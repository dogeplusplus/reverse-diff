use std::ops::{Add, Sub, Div, Mul};
use std::collections::{HashMap};
use std::hash::{Hash, Hasher};

#[derive(Clone,Debug)]
struct Variable {
    name: String,
    data: f32,
    grad: Vec<(Variable, f32)>,
}

impl Variable {
    fn new(name: String, data: f32) -> Self {
        Variable {
            name: name,
            data: data,
            grad: vec![],
        }
    }
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

impl Eq for Variable {}

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
            grad: vec![(self.clone(), other.data), (other, self.data)],
        }
    }
}

impl Div for Variable {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let x = self.data;
        let y = other.data;

        Self {
            name: format!("{} / {}", self.name, other.name),
            data: self.data * other.data,
            grad: vec![(self.clone(), 1. / y), (other, - x / y.powf(2.))],
        }
    }
}

fn compute_gradients(gradients: &mut HashMap<Variable, f32>, var: Variable, current_value: f32) {
    for tup in var.grad {
        let child = tup.0;
        let child_clone = child.clone();
        let gradient = tup.1;
        let value_to_child = current_value * gradient;
        if gradients.contains_key(&child) {
            let child_grad = gradients.entry(child).or_insert(0.);
            *child_grad += value_to_child;
        } else {
            gradients.insert(child, value_to_child);
        }

        compute_gradients(gradients, child_clone, value_to_child);
    }
}

fn get_gradients(var: Variable) -> HashMap<Variable, f32> {
    let mut gradients = HashMap::new();
    compute_gradients(&mut gradients, var, 1.);
    gradients
}

fn main() {
    let a = Variable::new("a".to_string(), 4.);
    let b = Variable::new("b".to_string(), 3.);
    let c = a.clone() + b;
    let d = a.clone() * c;
    println!("d: {:?}", d);

    let grad_d = get_gradients(d);
    println!("dd/da: {}", grad_d[&a]);

}
