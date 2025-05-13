#![allow(unused)]

use std::f32::consts::PI;
use std::fs;
use std::time::{Duration, Instant};
use neurox::{Executor, Optimizer};
use neurox::activation::Activation::{Linear, PNSigmoid, ReLU, Sigmoid, TanH};
use neurox::dual_vec::DualVec;
use neurox::Executor::CPU;
use neurox::layer::LayerType::Dense;
use neurox::loss::Loss;
use neurox::network::Network;

pub fn main() {
    let gpu = &Executor::gpu();
    // let layers = vec![
    //     // (&CPU, Dense(256, TanH)),
    //     (&CPU, Dense(32, TanH)),
    //     (&CPU, Dense(32, TanH)),
    //     (&CPU, Dense(32, TanH)),
    //     (&CPU, Dense(1, Linear)),
    // ];
    // let mut network = Network::new(1, &layers).unwrap();
    let bytes = fs::read("test.neurox").unwrap();
    let mut network = Network::from_bytes(Some(gpu), bytes);

    let mut inputs = vec![];
    let mut targets = vec![];
    let samples = 1000;
    for i in 0..samples {
        let v = (i as f32 / samples as f32) * 6. - 3.;
        inputs.push(v);
        targets.push((v* 2. * PI).sin());
    }

    let mut inputs = DualVec::from_vec((&CPU, &CPU), inputs);

    // let st = Instant::now();
    // network.predict(&mut inputs);
    // let d = st.elapsed();
    // println!("{:?}", d);

    let mut s = Duration::ZERO;
    for i in 0..100 {
        let st = Instant::now();
        network.predict(&mut inputs);
        let d = st.elapsed();
        s += d;
    }
    println!("{:?}", s);

    // let mut targets = DualVec::from_vec((&CPU, &CPU), targets);
    // let mut outputs = network.predict(&mut inputs);
    // if let (Some(inputs), Some(targets), Some(outputs)) = (inputs.cpu_borrow(), targets.cpu_borrow(), outputs.cpu_borrow()) {
    //     for i in 0..samples {
    //         println!("B in {} target {} out {}", inputs[i], targets[i], outputs[i]);
    //     }
    // }
    //
    // let res = network.train(&mut inputs, &mut targets, Optimizer::GradientDecent(0.02), Loss::MeanSquared, 10000, 4)
    //     .inspect_err(|e| {
    //         println!("{}", e)
    //     }
    //     );
    //
    // let mut outputs = network.predict(&mut inputs);
    // if let (Some(inputs), Some(targets), Some(outputs)) = (inputs.cpu_borrow(), targets.cpu_borrow(), outputs.cpu_borrow()) {
    //     for i in 0..samples {
    //         if i % 2 == 0 {
    //             println!("{},{},{}", inputs[i], targets[i], outputs[i]);
    //         }
    //     }
    // }
}