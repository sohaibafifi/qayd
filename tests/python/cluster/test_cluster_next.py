"""Exercise bounded continuation, immutable build use and resubmission discovery."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bench"))
from cluster.lib import controller as launch


def test_gpu_filter_excludes_the_complete_host_and_fails_closed():
    data = {"resources": [{"network_address": "cpu1", "gpumodel": None},
                           {"network_address": "gpu1", "gpumodel": None},
                           {"network_address": "gpu1", "gpumodel": "H100"}]}
    assert "network_address NOT IN" in launch.cpu_property(data)
    assert "gpumodel IS NOT NULL" in launch.cpu_property(data)
    with pytest.raises(ValueError, match="GPU-host exclusion"):
        launch.cpu_property({"resources": [{"network_address": "unknown"}]})


def test_worker_capacity_obeys_cores_memory_and_requested_workers():
    assert launch.worker_count("auto", 16384, 1, list(range(48)), 128 * 1024) == 7
    assert launch.worker_count("auto", 1024, 2, list(range(5)), 128 * 1024) == 2
    assert launch.worker_count("auto", 1024, 1, list(range(64)), 128 * 1024) == 16
    with pytest.raises(ValueError, match="insufficient allocation"):
        launch.worker_count("8", 16384, 1, list(range(48)), 128 * 1024)


def settings(tmp_path):
    return {"out": str(tmp_path), "root": str(tmp_path), "python": "/venv/bin/python",
            "driver": "/shared/driver.py", "prefix": "qayd-test-", "resource": "/nodes=1",
            "walltime": "12:00:00", "property": "cpu_only = 1", "jobs": 2,
            "profiles": {"main": {"config": "unused"}}, "files": {}, "pending": []}


def test_submission_has_required_types_cpu_predicate_and_no_nested_oar(tmp_path):
    saved = settings(tmp_path)
    tasks = tmp_path / "tasks list.txt"
    command = launch.command_for(saved, tmp_path / "plan.json", tasks, "1-0")
    assert [command[i + 1] for i, value in enumerate(command[:-1]) if value == "-t"] == ["idempotent", "besteffort"]
    assert command[command.index("-p") + 1] == saved["property"]
    assert command[-1].count("--worker") == 1
    assert "oarsub" not in command[-1]
    assert "--plan" in command[-1] and "tasks list.txt" in command[-1]


def test_idempotent_child_is_discovered_even_when_the_original_job_ended(monkeypatch):
    response = {"20": {"state": "Error", "name": "qayd-test-0"},
                "21": {"state": "Waiting", "name": "qayd-test-0", "resubmit_job_id": 20}}
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps(response))
    monkeypatch.setattr(launch, "run", fake)
    assert launch.live_jobs("qayd-test-") == [response["21"]]
    assert "job_name LIKE 'qayd-test-%'" in calls[0]


def test_existing_live_pool_never_causes_an_extra_submission(tmp_path, monkeypatch):
    saved = settings(tmp_path)
    monkeypatch.setattr(launch, "live_jobs", lambda _: [{"state": "Running"}])
    monkeypatch.setattr(launch, "next_stage", lambda *_: pytest.fail("live pool must be waited for"))
    monkeypatch.setattr(launch, "run", lambda *_a, **_k: pytest.fail("no extra submission"))
    launch.control(SimpleNamespace(out=tmp_path, execute=False), saved)
    assert launch.read(tmp_path / "state.json")["status"] == "running"


def test_audit_failure_blocks_further_oar_submissions(tmp_path, monkeypatch):
    saved = settings(tmp_path)
    monkeypatch.setattr(launch, "live_jobs", lambda _: [])
    def reject(*_):
        raise ValueError("gate audit failed")
    monkeypatch.setattr(launch, "next_stage", reject)
    monkeypatch.setattr(launch, "run", lambda *_a, **_k: pytest.fail("failed gate must not submit"))
    with pytest.raises(ValueError, match="gate audit failed"):
        launch.control(SimpleNamespace(out=tmp_path, execute=True), saved)


def test_oar_resubmission_quiet_period_prevents_a_racing_duplicate(tmp_path, monkeypatch):
    saved = settings(tmp_path)
    launch.write(tmp_path / "state.json", {"waves": 1, "inflight": True, "empty_checks": 0})
    monkeypatch.setattr(launch, "live_jobs", lambda _: [])
    monkeypatch.setattr(launch, "next_stage", lambda *_: pytest.fail("wait before resubmission"))
    launch.control(SimpleNamespace(out=tmp_path, execute=False), saved)
    assert launch.read(tmp_path / "state.json")["empty_checks"] == 1


def test_java_is_copied_with_libraries_and_the_shared_copy_is_checked(tmp_path, monkeypatch):
    original = tmp_path / "system-java"
    (original / "bin").mkdir(parents=True)
    (original / "lib").mkdir()
    (original / "bin/java").write_text("java executable")
    (original / "lib/modules").write_bytes(b"JVM modules")
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="", stderr=f"    java.home = {original}\n")
    monkeypatch.setattr(launch, "run", fake)
    executable, _files = launch.shared_java(tmp_path, "java")
    assert Path(executable).read_text() == "java executable"
    original.joinpath("lib/modules").write_bytes(b"system updated")
    assert launch.shared_java(tmp_path, "java")[0] == executable
    Path(executable).parent.parent.joinpath("lib/modules").write_bytes(b"shared updated")
    with pytest.raises(ValueError, match="runtime changed"):
        launch.shared_java(tmp_path, "java")
    assert len(calls) == 2


def test_adding_lkh_only_opens_its_own_vrptw_profile(tmp_path):
    saved = settings(tmp_path)
    config = tmp_path / "base.json"
    launch.write(config, {"datasets": [{"track": "cvrp"}, {"track": "cvrptw", "expected_count": 356}]})
    lkh = tmp_path / "LKH"
    lkh.write_text("binary"); lkh.chmod(0o755)
    saved.update(base_config=str(config), lkh=None, pending=["VRPTW pending"])
    launch.add_vrptw(saved, str(lkh))
    assert saved["profiles"]["main"] == {"config": "unused"}
    assert launch.read(saved["profiles"]["vrptw"]["config"])["datasets"] == [{"track": "cvrptw", "expected_count": 356}]
    assert not saved["pending"]


def test_preflight_preserves_the_validated_source_and_build_receipt(tmp_path, monkeypatch):
    from cluster.lib import campaign, report
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    monkeypatch.setattr(launch, "modules", lambda _: (campaign, report))
    (tmp_path / "Cargo.toml").write_text("[workspace]\n")
    script = tmp_path / "bench/cluster/lib/campaign.py"
    script.parent.mkdir(parents=True)
    script.write_text("# deployed frozen source\n")
    config_path = tmp_path / "bench/cluster/runs/corpus-ready/config.json"
    config = {"schema": "qayd.cluster.config/v1", "name": "validated", "seeds": [0],
              "memory_mb": 16384, "checker_memory_mb": 4096, "checker_seconds": 120, "grace_seconds": 2,
              "protocols": {"cvrp": {"seconds": 1}, "cvrptw": {"seconds": 1}},
              "solvers": {"cvrp": ["qayd-native", "hgs"], "cvrptw": ["qayd-native", "lkh"]},
              "datasets": [{"track": "cvrp", "family": "x", "glob": "none"},
                           {"track": "cvrptw", "family": "solomon", "glob": "none"}]}
    launch.write(config_path, config)
    binary = tmp_path / "target/release/qayd"
    binary.parent.mkdir(parents=True); binary.write_bytes(b"validated LP binary")
    source_before = campaign.source_identity()["sha256"]
    receipt_path = tmp_path / "bench/cluster/runs/build.json"
    launch.write(receipt_path, {"source_sha256": source_before, "features": {
        "binaries": ["lp-relaxation"], "python": ["python", "lp-relaxation"]},
        "artifacts": {str(binary): launch.sha(binary)}})
    receipt_before = receipt_path.read_bytes()
    hgs = tmp_path / "bench/cluster/runs/tools/solvers/HGS-CVRP/build/hgs"
    hgs.parent.mkdir(parents=True); hgs.write_text("hgs")
    hgs.chmod(0o755)
    manifest = tmp_path / "java.json"
    launch.write(manifest, {})
    monkeypatch.setattr(launch, "shared_java", lambda *_: (str(tmp_path / "java/bin/java"), {str(manifest): launch.sha(manifest)}))
    out = tmp_path / "bench/cluster/runs/next"
    out.mkdir()
    args = SimpleNamespace(config=None, property="cpu_only=1", java="java", hgs=None, lkh=None,
                           jobs=2, workers="auto", resource="/nodes=1", walltime="12:00:00", queue=None, hexaly=False)
    monkeypatch.delenv("LKH_BINARY", raising=False)
    monkeypatch.delenv("HGS_BINARY", raising=False)
    saved = launch.initial_settings(args, tmp_path, out)
    assert campaign.source_identity()["sha256"] == source_before
    assert receipt_path.read_bytes() == receipt_before
    assert launch.read(saved["profiles"]["main"]["config"])["solvers"]["cvrp"] == ["qayd-native", "hgs"]
    assert [d["track"] for d in launch.read(saved["profiles"]["main"]["config"])["datasets"]] == ["cvrp"]
    assert saved["pending"] and saved["lkh"] is None


def test_automatic_smoke_gate_full_sequence_is_bounded_and_requires_a_clean_gate(tmp_path, monkeypatch):
    from cluster.lib import campaign, report
    saved = settings(tmp_path)
    saved.update(java="/shared/java", hgs="/shared/hgs", lkh=None)
    monkeypatch.setattr(launch, "modules", lambda _: (campaign, report))
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    monkeypatch.setattr(campaign, "verify_snapshot", lambda _plan: None)
    monkeypatch.setattr(launch, "live_jobs", lambda _: [])
    monkeypatch.setattr(launch.time, "sleep", lambda _: None)
    submissions = []
    preparations = []

    def fake_run(command, **kwargs):
        command = list(map(str, command))
        if command[0] == "oarsub":
            submissions.append(command)
            import shlex
            payload = shlex.split(command[-1])
            plan_path = Path(payload[payload.index("--plan") + 1])
            taskfile = Path(payload[payload.index("--tasks") + 1])
            plan = campaign.load_plan(plan_path)
            for index in map(int, taskfile.read_text().split()):
                task = plan["tasks"][index]
                arm = {"solver": "qayd", "status": "OPTIMUM", "objective": 1,
                       "has_incumbent": True, "solved": True, "validation": {"valid": True},
                       "wall_seconds": 1, "par2": 1, "details": {}, "host": {"hostname": "cpu"},
                       "plan_sha256": plan["sha256"], "task_key": task["key"]}
                campaign.write(plan_path.parent / "tasks" / f"{index:07d}" / "result.json", {
                    "complete": True, "plan_sha256": plan["sha256"], "task_key": task["key"], "arms": [arm]})
        elif "prepare" in command:
            stage = command[command.index("--stage") + 1]
            preparations.append(stage)
            directory = Path(command[command.index("--out") + 1])
            if stage == "full":
                assert read_gate(command)["ready"] is True
            tasks = [{"key": str(i), "solvers": ["qayd"], "seed": 0, "seconds": 1,
                      "item": {"id": str(i), "track": "xcsp-cop", "family": "fixture", "sense": "min"}}
                     for i in range(3)]
            plan = {"root": str(tmp_path), "tasks": tasks, "stage": stage, "environment": "same-environment"}
            plan["sha256"] = campaign.digest(plan)
            campaign.write(directory / "plan.json", plan)
        else:
            pytest.fail(f"unexpected command: {command}")
        return SimpleNamespace(returncode=0)

    def read_gate(command):
        return launch.read(command[command.index("--gate") + 1])

    monkeypatch.setattr(launch, "run", fake_run)
    launch.control(SimpleNamespace(out=tmp_path, execute=True, poll_seconds=5), saved)
    assert preparations == ["smoke", "gate", "full"]
    assert len(submissions) == 6
    assert launch.read(tmp_path / "state.json")["status"] == "complete"
    for wave in range(3):
        groups = [p.read_text().split() for p in (tmp_path / "waves" / f"{wave:05d}").glob("group-*.txt")]
        assert len(groups) == 2
        assert sorted(int(i) for group in groups for i in group) == [0, 1, 2]


def test_unrecognized_oarstat_response_does_not_look_like_an_empty_queue(monkeypatch):
    monkeypatch.setattr(launch, "run", lambda *_a, **_k: SimpleNamespace(stdout='{"error":"unavailable"}'))
    with pytest.raises(ValueError, match="unexpected oarstat"):
        launch.live_jobs("qayd-test-")


def test_a_failed_independent_profile_does_not_block_a_qualified_profile(tmp_path, monkeypatch):
    from cluster.lib import campaign, report
    saved = settings(tmp_path)
    saved['profiles']['hexaly'] = {'config': 'unused'}
    monkeypatch.setattr(launch, 'live_jobs', lambda _: [])
    monkeypatch.setattr(launch, 'modules', lambda _: (campaign, report))
    def candidate(_settings, profile):
        if profile == 'hexaly':
            raise ValueError('large JSSP gate needs investigation')
        return 'gate', tmp_path / 'main/gate/plan.json', {
            'tasks': [{'seconds': 10, 'solvers': ['qayd']}]
        }, [0]
    monkeypatch.setattr(launch, 'next_stage', candidate)
    launch.control(SimpleNamespace(out=tmp_path, execute=False), saved)
    state = launch.read(tmp_path / 'state.json')
    assert state['profile'] == 'main'
    assert 'hexaly' in state['profile_errors']
    assert list((tmp_path / 'waves/00000').glob('group-*.txt'))


def test_hexaly_reference_retry_preserves_the_created_config_and_completed_pages(tmp_path, monkeypatch):
    saved = settings(tmp_path)
    config = tmp_path / 'hexaly-reference/config.json'
    saved['profiles']['hexaly'] = {'config': str(config)}
    launch.write(config, {'published_reference': {'pages': {
        'cvrp': 'https://example.invalid/cvrp', 'jssp': 'https://example.invalid/jssp'}}})
    original = config.read_bytes()
    first = config.parent / 'cvrp.html'
    first.write_text('already archived')
    calls = []
    def fake(command, **kwargs):
        assert command[0] == 'curl'
        calls.append(command[-1])
        Path(command[command.index('--output') + 1]).write_text('archived jssp')
    monkeypatch.setattr(launch, 'run', fake)
    launch.prepare_hexaly(saved)
    launch.prepare_hexaly(saved)
    assert calls == ['https://example.invalid/jssp']
    assert config.read_bytes() == original
    assert first.read_text() == 'already archived'
    assert set(launch.read(config.parent / 'sources.json')) == {'cvrp', 'jssp'}


@pytest.mark.parametrize('output', ['', '\n', '  \n\t'])
def test_successful_empty_oarstat_is_an_empty_job_list(tmp_path, monkeypatch, output):
    cli = tmp_path / 'oarstat'
    cli.write_text('#!/usr/bin/env python3\nimport sys\nsys.stdout.write(' + repr(output) + ')\n')
    cli.chmod(0o755)
    monkeypatch.setenv('PATH', str(tmp_path) + os.pathsep + os.environ['PATH'])
    assert launch.live_jobs('qayd-never-submitted-') == []


def test_empty_oarstat_with_an_error_does_not_look_like_an_empty_queue(monkeypatch):
    monkeypatch.setattr(launch, 'run', lambda *_a, **_k: SimpleNamespace(stdout='', stderr='database unavailable'))
    with pytest.raises(ValueError, match='oarstat.*database unavailable'):
        launch.live_jobs('qayd-test-')


def test_failed_oarstat_is_not_accepted_as_an_empty_queue(tmp_path, monkeypatch):
    cli = tmp_path / 'oarstat'
    cli.write_text('#!/usr/bin/env python3\nraise SystemExit(2)\n')
    cli.chmod(0o755)
    monkeypatch.setenv('PATH', str(tmp_path) + os.pathsep + os.environ['PATH'])
    with pytest.raises(subprocess.CalledProcessError):
        launch.live_jobs('qayd-test-')


def test_non_json_oarstat_reports_the_command_output(monkeypatch):
    monkeypatch.setattr(launch, 'run', lambda *_a, **_k: SimpleNamespace(stdout='OAR service unavailable', stderr='try later'))
    with pytest.raises(ValueError, match='oarstat.*OAR service unavailable.*try later'):
        launch.live_jobs('qayd-test-')


def test_a_corrupt_json_file_is_identified_by_its_path(tmp_path):
    path = tmp_path / 'shared-java.json'
    path.write_text('')
    with pytest.raises(ValueError, match=r'invalid JSON in .*shared-java.json'):
        launch.read(path)


def test_fixed_controller_resumes_old_settings_without_rewriting_the_frozen_driver(tmp_path, monkeypatch):
    out = tmp_path / 'bench/cluster/runs/next'
    out.mkdir(parents=True)
    old_driver = out / 'driver.py'
    old_driver.write_text('# previously frozen worker\n')
    saved = settings(out)
    saved.update(root=str(tmp_path), driver=str(old_driver), files={str(old_driver): launch.sha(old_driver)})
    launch.write(out / 'settings.json', saved)
    launch.write(out / 'state.json', {'waves': 0, 'empty_checks': 0, 'status': 'failed'})
    receipt = out.parent / 'build.json'
    receipt.write_bytes(b'validated build receipt')
    before_driver, before_receipt = old_driver.read_bytes(), receipt.read_bytes()
    cli = tmp_path / 'oarstat'
    cli.write_text('#!/usr/bin/env python3\n')
    cli.chmod(0o755)
    monkeypatch.setenv('PATH', str(tmp_path) + os.pathsep + os.environ['PATH'])
    monkeypatch.setattr(launch, 'initial_settings', lambda *_: pytest.fail('must preserve existing settings'))
    monkeypatch.setattr(launch, 'next_stage', lambda *_: None)
    monkeypatch.setattr(sys, 'argv', [launch.__file__, '--repo', str(tmp_path), '--out', str(out)])
    launch.main()
    assert launch.read(out / 'state.json')['status'] == 'complete'
    assert old_driver.read_bytes() == before_driver
    assert receipt.read_bytes() == before_receipt
    assert launch.read(out / 'settings.json')['files'][str(old_driver)] == saved['files'][str(old_driver)]


def test_deadline_budget_subtracts_preparation_and_publication(monkeypatch):
    monkeypatch.setattr(launch.time, 'monotonic', lambda: 104)
    assert launch.remaining_search(10, 100) == 5.5
    assert launch.remaining_search(600, 100) == 566
    assert launch.remaining_search(2, 100) == 0


def test_deadline_without_output_is_unknown_and_never_creates_a_witness(tmp_path):
    stdout, stderr = tmp_path/'stdout', tmp_path/'stderr'
    stdout.write_text(''); stderr.write_text('')
    record = {'status': 'ERROR', 'timed_out': True, 'returncode': -15, 'has_incumbent': False,
              'objective': None, 'proof_claimed': False, 'validation': {'valid': None},
              'details': {'parse_error': 'solver did not emit a JSON object'},
              'stdout': str(stdout), 'stderr': str(stderr)}
    result = launch.normalize_deadline(dict(record))
    assert result['status'] == 'UNKNOWN' and result['raw_status'] == 'ERROR'
    assert result['objective'] is None and result['proof_claimed'] is False
    for changes in ({'returncode': 1}, {'timed_out': False}, {'validation': {'valid': False}}, {'status': 'INVALID'}):
        result = launch.normalize_deadline({**record, **changes})
        assert result['status'] == changes.get('status', 'ERROR')
    stdout.write_text('{malformed JSON')
    assert launch.normalize_deadline(dict(record))['status'] == 'ERROR'


def test_qayd_adapter_accounts_for_preparation_before_model_solve(tmp_path, monkeypatch):
    import qayd
    calls = []
    monkeypatch.setattr(qayd.Model, 'solve', lambda _self, **kw: calls.append(kw))
    clock = iter([100, 104])
    monkeypatch.setattr(launch.time, 'monotonic', lambda: next(clock))
    script = tmp_path/'model.py'
    script.write_text('import qayd\nqayd.Model().solve(time_limit=10, seed=5)\n')
    launch.model_adapter(['--solver', 'qayd-native', '--seconds', '10', '--script', str(script), '--'])
    assert calls == [{'time_limit': 5, 'seed': 5}]


def test_ortools_adapter_accounts_for_preparation_before_search(tmp_path, monkeypatch):
    from ortools.sat.python import cp_model
    calls = []
    monkeypatch.setattr(cp_model.CpSolver, 'solve', lambda solver, *_: calls.append(solver.parameters.max_time_in_seconds))
    clock = iter([100, 104])
    monkeypatch.setattr(launch.time, 'monotonic', lambda: next(clock))
    script = tmp_path/'model.py'
    script.write_text('from ortools.sat.python import cp_model\ns=cp_model.CpSolver()\ns.parameters.max_time_in_seconds=10\ns.solve(None)\n')
    launch.model_adapter(['--solver', 'ortools-cp-sat', '--seconds', '10', '--script', str(script), '--'])
    assert calls == [5]


def test_v2_plans_preserve_templates_and_require_a_matching_gate_before_full(tmp_path, monkeypatch):
    from cluster.lib import campaign, report
    monkeypatch.setattr(campaign, 'ROOT', tmp_path)
    monkeypatch.setattr(launch, 'modules', lambda _: (campaign, report))
    driver=tmp_path/'driver.py'; driver.write_text('# deadline wrapper')
    saved=settings(tmp_path)
    saved.update(driver=str(driver), java='/java', hgs='/hgs', lkh=None, execution_policy=launch.EXECUTION_POLICY)
    config={'protocols': {'jssp': {'seconds': 120}}, 'seeds': [0,1], 'solvers': {'jssp': ['qayd-native']}}
    item={'track':'jssp','path':'instance','id':'one','family':'jssp','sense':'min'}
    assets={'instance':'digest'}
    originals={}
    def prepare(command, **kwargs):
        directory=Path(command[command.index('--out')+1])
        stage=command[command.index('--stage')+1]
        task={'item':item,'seed':0,'seconds':2 if stage=='smoke' else 10,'solvers':['qayd-native']}
        task['key']=campaign.digest(task)
        plan={'root':str(tmp_path),'stage':stage,'environment':'base','artifacts':{},'tasks':[task],
              'overview':{},'assets':assets,'config':config,'source':{'sha256':'validated-source'}}
        plan['sha256']=campaign.digest(plan)
        campaign.write(directory/'plan.json',plan)
        originals[directory/'plan.json']=(directory/'plan.json').read_bytes()
    monkeypatch.setattr(launch,'run',prepare)
    monkeypatch.setattr(campaign,'discover',lambda _: ([item],assets,[]))
    smoke=tmp_path/'main/smoke'
    launch.freeze_deadline_plan(saved,'main','smoke',smoke)
    smoke_plan=campaign.load_plan(smoke/'plan.json')
    assert smoke_plan['tasks'][0]['seconds']==30
    assert smoke_plan['source']=={'sha256':'validated-source'}
    gate=tmp_path/'main/gate'
    launch.freeze_deadline_plan(saved,'main','gate',gate)
    gate_plan=campaign.load_plan(gate/'plan.json')
    campaign.write(gate/'qualification.json',{'ready':False})
    with pytest.raises(ValueError,match='requires a clean v2 gate'):
        launch.freeze_deadline_plan(saved,'main','full',tmp_path/'main/full')
    campaign.write(gate/'qualification.json',{'ready':True,'stage':'gate','environment':gate_plan['environment'],
                                              'plan_sha256':gate_plan['sha256']})
    launch.freeze_deadline_plan(saved,'main','full',tmp_path/'main/full')
    full=campaign.load_plan(tmp_path/'main/full/plan.json')
    assert full['environment']==gate_plan['environment']
    assert [task['seconds'] for task in full['tasks']]==[120,120]
    assert sorted(task['seed'] for task in full['tasks'])==[0,1]
    assert full['artifacts'][str(driver)]==launch.sha(driver)
    assert all(path.read_bytes()==data for path,data in originals.items())


def test_bundled_minizinc_is_found_without_shell_activation(tmp_path,monkeypatch):
    bundle=tmp_path/'bench/cluster/runs/tools/MiniZincIDE-2.9.5-bundle-linux-x86_64/bin/minizinc'
    bundle.parent.mkdir(parents=True); bundle.write_text('#!/bin/sh\n'); bundle.chmod(0o755)
    monkeypatch.setenv('PATH','/no-tools')
    assert launch.find_minizinc(tmp_path)==str(bundle)
