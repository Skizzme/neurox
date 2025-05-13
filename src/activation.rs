
#[derive(Clone, Debug)]
pub enum Activation {
    Linear,
    ReLU,
    TanH,
    Sigmoid,
    PNSigmoid
}

impl Activation {
    pub fn activate(&self, x: f32) -> f32 {
        match self {
            Activation::Linear => x,
            Activation::ReLU => if x > 0. { x } else { 0. },
            Activation::TanH => (x.exp() - (-x).exp()) / (x.exp() + (-x).exp()),
            Activation::Sigmoid => 1. / (1. + (-x).exp()),
            Activation::PNSigmoid => 2. / (1. + (-x).exp()) - 1.,
        }
    }

    pub fn derivative(&self, x: f32) -> f32 {
        match self {
            Activation::Linear => 1.,
            Activation::ReLU => if x > 0. { 1. } else { 0. },
            Activation::TanH => {
                let b = self.activate(x);
                1. - (b * b)
            }
            Activation::Sigmoid => {
                let b = self.activate(x);
                b * (1. - b)
            }
            Activation::PNSigmoid => {
                let ex = x.exp();
                (2. * ex) / (ex + 1.).powf(2.)
            }
        }
    }
}

impl From<usize> for Activation {
    fn from(value: usize) -> Self {
        match value {
            1 => Activation::ReLU,
            2 => Activation::TanH,
            3 => Activation::Sigmoid,
            4 => Activation::PNSigmoid,
            _ => Activation::Linear,
        }
    }
}

impl Into<usize> for &Activation {
    fn into(self) -> usize {
        match self {
            Activation::Linear => 0,
            Activation::ReLU => 1,
            Activation::TanH => 2,
            Activation::Sigmoid => 3,
            Activation::PNSigmoid => 4,
        }
    }
}