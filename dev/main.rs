#![allow(unused)]

use neurox::{Executor, Optimizer};
use neurox::activation::Activation::{Linear, Sigmoid, TanH};
use neurox::dual_vec::DualVec;
use neurox::Executor::CPU;
use neurox::layer::LayerType::Dense;
use neurox::loss::Loss;
use neurox::network::Network;

pub fn main() {
    let gpu = &Executor::gpu();
    let layers = vec![
        // (&CPU, Dense(4, Linear)),
        (&CPU, Dense(1, Sigmoid)),
    ];
    let mut network = Network::new(2, &layers).unwrap();

    let mut inputs = DualVec::from_vec((&CPU, &CPU), vec![-0.6, 0.5, 0.6, 0.5, 0., 0., 1., 1.]);
    let mut targets = DualVec::from_vec((&CPU, &CPU), vec![0., 1., 1., 0.,]);
    let res = network.train(&mut inputs, &mut targets, Optimizer::GradientDecent(0.02), Loss::MeanSquared, 1, 2)
        .inspect_err(|e| {
            println!("{}", e)
        }
        );
}