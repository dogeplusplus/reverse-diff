#[cfg(test)]
mod tests {
    use reverse_diff::{Variable, get_gradients, sin, cos, exp, ln};

    #[test]
    fn variable_instantiate() {
        let a = Variable::new("a".to_string(), 4.);
        let b = Variable::new("a".to_string(), 4.);
        assert_eq!(a, b);
    }

    #[test]
    fn variable_add() {
        let a = Variable::new("a".to_string(), 4.);
        let b = Variable::new("b".to_string(), 3.);
        let c = a + b;
        assert_eq!(c.data, 7.);
    }

    #[test]
    fn variable_mul() {
        let a = Variable::new("a".to_string(), 4.);
        let b = Variable::new("b".to_string(), 3.);
        let c = a * b;
        assert_eq!(c.data, 12.);
    }

    #[test]
    fn variable_sub() {
        let a = Variable::new("a".to_string(), 4.);
        let b = Variable::new("b".to_string(), 3.);
        let c = a - b;
        assert_eq!(c.data, 1.);
    }

    #[test]
    fn variable_div() {
        let a = Variable::new("a".to_string(), 4.);
        let b = Variable::new("b".to_string(), 2.);
        let c = a / b;
        assert_eq!(c.data, 2.);
    }

    #[test]
    fn gradient_calculation() {
        let a = Variable::new("a".to_string(), 4.);
        let b = Variable::new("b".to_string(), 3.);
        let c = a.clone() + b.clone();
        let d = a.clone() * c.clone();
        let grad_d = get_gradients(d);

        assert_eq!(grad_d[&a], 11.0);
        assert_eq!(grad_d[&b], 4.0);
        assert_eq!(grad_d[&c], 4.0);
    }

    #[test]
    fn trigonometric_functions() {
        let a = Variable::new("a".to_string(), std::f32::consts::PI / 2.);
        let sin_a = sin(&a);
        let cos_a = cos(&a);
        assert!((sin_a.data - 1.).abs() < f32::EPSILON);
        assert!((cos_a.data - 0.).abs() < f32::EPSILON);

        let b = Variable::new("b".to_string(), 1.);
        let exp_b = exp(&b);
        let log_b = ln(&b);
        assert!((exp_b.data - std::f32::consts::E).abs() < f32::EPSILON);
        assert!((log_b.data - 0.).abs() < f32::EPSILON);
    }
}
