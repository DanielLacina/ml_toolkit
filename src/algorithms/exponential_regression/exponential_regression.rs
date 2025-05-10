use super::utils::*;
use crate::linear_algebra::{Matrix, RowVector};
use std::iter::zip;

pub struct ExponentialRegression {
    c: f32,
    k: f32,
}

impl ExponentialRegression {
    pub fn new() -> Self {
        return Self { c: 0.0, k: 0.0 };
    }

    pub fn fit(&mut self, data: &Vec<f32>, labels: &Vec<f32>, k: f32, c: f32) {
        // 1st parameter - k (exponent)
        // 2nd parameter - c (scalar)
        // y = Ce^-kx
        let mut parameters = vec![k, c];
        let converging_tolerance = (10.0 as f32).powf(-9.0);
        loop {
            let [k, c] = [parameters[0], parameters[1]];
            let predictions: Vec<f32> = data.iter().map(|x_value| exp_x(*x_value, c, k)).collect();
            let residuals: Matrix = Matrix::new(
                &zip(labels, predictions.iter())
                    .map(|(label, prediction)| RowVector::new(&vec![*label - *prediction]))
                    .collect::<Vec<RowVector>>(),
            );
            let residual_p_k: RowVector = RowVector::new(
                &data
                    .iter()
                    .map(|x_value| residual_p_k(*x_value, c, k))
                    .collect(),
            );
            let residual_p_c: RowVector = RowVector::new(
                &data
                    .iter()
                    .map(|x_value| residual_p_c(*x_value, k))
                    .collect(),
            );
            let jacobian_transposed = Matrix::new(&vec![residual_p_k, residual_p_c]);
            let jacobian = jacobian_transposed.transpose();
            let jacobian_dotted = jacobian_transposed.multiply(&jacobian);
            let jacobian_dotted_inverse = jacobian_dotted.inverse();
            let residual_dotted = jacobian_transposed.multiply(&residuals);
            let gradient = jacobian_dotted_inverse.multiply(&residual_dotted);
            println!("{:?}", gradient.transpose().get(0));
            parameters = zip(parameters, gradient.matrix())
                .map(|(parameter, gradient_value)| {
                    let gradient_value = gradient_value.get(0);
                    parameter - gradient_value
                })
                .collect();
        }
        self.k = parameters[0];
        self.c = parameters[1];
    }

    pub fn predict(&self, data: &Vec<f32>) -> Vec<f32> {
        data.iter()
            .map(|x_value| exp_x(*x_value, self.c, self.k))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_exponential_regression() {
        let labels = vec![2.0, 4.0, 8.0, 16.0, 24.0];
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut exponential_regression = ExponentialRegression::new();
        exponential_regression.fit(&features, &labels, 1.0, 1.0);
        println!("{:?}", exponential_regression.predict(&features));
    }
}
