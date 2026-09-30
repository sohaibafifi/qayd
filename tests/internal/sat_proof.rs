use crate::engines::sat::proof::{ProofFormat, ProofStep, ProofWriter};

#[test]
fn drat_writer_serializes_add_delete_and_empty_clause() {
    let mut writer = ProofWriter::new(Vec::new(), ProofFormat::Drat);
    writer.write_step(&ProofStep::Add(vec![1, -2, 3])).unwrap();
    writer.write_step(&ProofStep::Delete(vec![-4])).unwrap();
    writer.write_step(&ProofStep::Add(Vec::new())).unwrap();
    let bytes = writer.into_inner();
    assert_eq!(String::from_utf8(bytes).unwrap(), "1 -2 3 0\nd -4 0\n0\n");
}
