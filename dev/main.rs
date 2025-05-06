use std::time::Instant;
use neurox::{Executor, Network};
use neurox::activation::Activation::TanH;
use neurox::dual_vec::DualVec;
use neurox::Executor::CPU;
use neurox::layer::LayerType::Dense;

pub fn main() {
    let gpu = &Executor::gpu();
    let mut network = Network::new(2, vec![
        // (gpu, Attention(6, 12, 20, 44)),
        // (gpu, Dense(46, ReLU)),
        // (&CPU, Dense(1024, TanH)),
        // (&CPU, Dense(512, TanH)),
        (&CPU, Dense(4, TanH)),
        (&CPU, Dense(2, TanH))
    ]);
    let mut input = DualVec::from_exec(&CPU, 2);
    input.cpu().unwrap().borrow_mut().append(&mut vec![0.2, 0.3]);

    let st = Instant::now();
    let mut output = network.predict(&mut input);
    let d = st.elapsed();
    println!("{:?} {:?}", output.cpu().unwrap(), d);
}