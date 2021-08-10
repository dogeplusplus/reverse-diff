use std::ops::{Add, Sub, Div, Mul};
use std::collections::{HashMap};
use std::hash::{Hash, Hasher};

#[derive(Clone,Eq)]
struct Variable {
    name: String,
    data: f32,
    grad: Vec<(Variable, f32)>,
}

impl Add for Variable {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            name: format!("{} + {}", self.name, other.name),
            data: self.data + other.data,
            grad: vec![(self, 1.), (other, 1.)],
        }
    }
}

impl PartialEq for Variable {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Hash for Variable {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl Sub for Variable {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            name: format!("{} - {}", self.name, other.name),
            data: self.data - other.data,
            grad: vec![(self, 1.), (other, -1.)]
        }
    }
}

impl Mul for Variable {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            name: format!("{} * {}", self.name, other.name),
            data: self.data * other.data,
            grad: vec![(self, other.data), (other, self.data)],
        }
    }
}

impl Div for Variable {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self {
            name: format!("{} / {}", self.name, other.name),
            data: self.data * other.data,
            grad: vec![(self, 1. / other.data), (other, - self.data / other.data.powf(2.))],
        }
    }
}

fn compute_gradients(var: Variable, current_value: f32) -> HashMap<Variable, Vec<Variable>> {
    let gradients = HashMap::new();

    for tup in var.grad {
        let child = tup.0;
        let gradient = tup.1;
        let value_to_child = current_value * gradient;
        gradients.insert(child, value_to_child);
    }

    gradients
}

fn get_gradients(var: Variable) -> HashMap<Variable, Vec<Variable>> {
    let gradients = HashMap::new();
}

fn main() {
    println!("Hello, world!");
}
