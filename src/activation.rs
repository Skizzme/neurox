
#[derive(Clone, Debug)]
pub enum Activation {
    Linear,
    ReLU,
    TanH,
}

impl Activation {
    pub fn activate(&self, x: f32) -> f32 {
        match self {
            Activation::Linear => x,
            Activation::ReLU => if x > 0. { x } else { 0. },
            Activation::TanH => (x.exp() - (-x).exp()) / (x.exp() + (-x).exp()),
        }
    }

    pub fn derivative(&self, x: f32) -> f32 {
        match self {
            Activation::Linear => 1.,
            Activation::ReLU => if x > 0. { 1. } else { 0. },
            Activation::TanH => {
                let b = self.activate(x);
                1.0 - (b * b)
            }
        }
    }
}