use crate::certificate::{hash::sha256, CertificateError};

#[test]
fn certificate_sha256_standard_vectors_and_padding_boundaries() {
    let vectors: &[(&[u8], &str)] = &[
        (b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        (b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"),
    ];
    for &(input, digest) in vectors {
        assert_eq!(sha256(input, &|| false).unwrap(), digest);
    }
    assert_eq!(sha256(&vec![b'a'; 1_000_000], &|| false).unwrap(), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

#[test]
fn certificate_sha256_observes_stop_before_and_during_blocks() {
    assert_eq!(sha256(b"", &|| true), Err(CertificateError::Interrupted));
    let calls = std::cell::Cell::new(0usize);
    let stop = || {
        calls.set(calls.get() + 1);
        calls.get() > 2
    };
    assert_eq!(sha256(&[42; 256], &stop), Err(CertificateError::Interrupted));
    assert_eq!(calls.get(), 3);
}
