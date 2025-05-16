#![allow(unused)]

use std::f32::consts::PI;
use std::fs;
use std::time::Instant;
use neurox::{Executor, Optimizer};
use neurox::activation::Activation::{Linear, PNSigmoid, ReLU, Sigmoid, TanH};
use neurox::dual_vec::DualVec;
use neurox::Executor::{CPU, GPU};
use neurox::layer::LayerType::Dense;
use neurox::loss::Loss;
use neurox::network::Network;

pub fn main() {
    let ex = &Executor::gpu();
    // let ex = &CPU;
    let layers = vec![
        (ex, Dense(256, TanH)),
        (ex, Dense(256, TanH)),
        (ex, Dense(64, TanH)),
        (ex, Dense(64, TanH)),
        (ex, Dense(16, TanH)),
        (ex, Dense(1, Linear)),
    ];
    let mut network = Network::new(1, &layers).unwrap();
    // let bytes = fs::read("test.neurox").unwrap();
    // let mut network = Network::from_bytes(Some(gpu), bytes);

    let mut inputs = vec![];
    let mut targets = vec![];
    let samples = 1000;
    for i in 0..samples {
        let v = (i as f32 / samples as f32);
        inputs.push(v);
        targets.push((v* 2. * PI).sin());
    }

    let mut inputs = DualVec::from_vec((&CPU, ex), inputs);
    let mut targets = DualVec::from_vec((&CPU, ex), targets);

    let res = network.train(&mut inputs, &mut targets, Optimizer::GradientDecent(0.02), Loss::MeanSquared, 400, 4)
        .inspect_err(|e| {
            println!("{}", e)
        }
        );

    let st = Instant::now();
    let mut outputs = network.predict(&mut inputs);
    let d = st.elapsed();
    if let (Some(inputs), Some(targets), Some(outputs)) = (inputs.cpu_borrow(), targets.cpu_borrow(), outputs.cpu_borrow()) {
        for i in 0..samples {
            if i % 10 == 0 {
                println!("{},{},{}", inputs[i], targets[i], outputs[i]);
            }
        }
    }
    println!("{:?}", d);

    let v = network.as_bytes();
}