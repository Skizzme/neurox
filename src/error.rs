
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("An error occurred when trying to access a buffer: {0}")]
    UnavailableBuffer(String),
    #[error("{0}")]
    Network(NetworkError),
    #[error("{0}")]
    Mismatch(MismatchError)
}

#[derive(Debug, thiserror::Error)]
pub enum MismatchError {
    #[error("Input sample count ({0}) does not match output sample count ({1})")]
    Sample(usize, usize),
}

#[derive(Debug, thiserror::Error)]
pub enum NetworkError {
    #[error("No layers were provided when creating the network")]
    ZeroLayers
}