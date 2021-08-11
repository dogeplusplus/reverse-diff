#[cfg(test)]
mod tests {
    use reverse_diff::{Variable, get_gradients};

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
}