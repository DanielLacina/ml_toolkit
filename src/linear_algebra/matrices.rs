use crate::linear_algebra::vectors::RowVector;

#[derive(Clone, PartialEq, Debug)]
pub struct Matrix {
    matrix: Vec<RowVector>,
}

impl Matrix {
    pub fn new(matrix: &Vec<RowVector>) -> Self {
        Self {
            matrix: matrix.clone(),
        }
    }

    pub fn to_matrix(multidim_vec: &Vec<Vec<f32>>) -> Self {
        Self::new(
            &multidim_vec
                .iter()
                .map(|vec_| RowVector::new(vec_))
                .collect::<Vec<RowVector>>(),
        )
    }

    pub fn transpose(&self) -> Matrix {
        let m = &self.matrix;
        let mut output_matrix = Vec::new();
        for i in 0..m[0].len() {
            let mut output_vector = Vec::new();
            for j in 0..m.len() {
                output_vector.push(m[j].get(i));
            }
            output_matrix.push(RowVector::new(&output_vector));
        }
        return Matrix::new(&output_matrix);
    }

    pub fn multiply(&self, m2: &Matrix) -> Matrix {
        let mut output_matrix = Vec::new();
        for i in 0..self.len() {
            let mut new_row = Vec::new();
            for j in 0..m2.get(0).len() {
                let mut sum = 0.0;
                for k in 0..m2.len() {
                    sum += self.get(i).get(k) * m2.get(k).get(j);
                }
                new_row.push(sum);
            }
            output_matrix.push(RowVector::new(&new_row));
        }
        return Matrix::new(&output_matrix);
    }

    pub fn len(&self) -> usize {
        self.matrix.len()
    }

    pub fn identity(&self) -> Matrix {
        let size = self.matrix.len();
        let mut i_matrix = Vec::new();
        for i in 0..size {
            let mut vector = Vec::new();
            for j in 0..size {
                if i == j {
                    vector.push(1.0);
                } else {
                    vector.push(0.0)
                }
            }
            i_matrix.push(RowVector::new(&vector));
        }
        return Matrix::new(&i_matrix);
    }

    pub fn inverse(&self) -> Matrix {
        let mut m_copy = self.matrix.clone();
        let mut identity_m = self.identity().matrix.clone();
        for i in 0..m_copy.len() {
            let mut pivet_row_i = i;
            while pivet_row_i < m_copy.len() && m_copy[pivet_row_i].get(i) == 0.0 {
                pivet_row_i += 1;
            }
            if pivet_row_i == m_copy.len() {
                panic!("cannot derive the identity matrix with a column of all zeros");
            }
            if pivet_row_i != i {
                self.swap_rows(&mut m_copy, pivet_row_i, i);
                self.swap_rows(&mut identity_m, pivet_row_i, i);
                // i now indexes the previous row indexed by pivet_row_i and vice versa
            }
            let pivet_value = m_copy[i].get(i);
            // sets the matrix[row_index][row_index] to 1
            m_copy[i] = m_copy[i].multiply_by_scalar(1.0 / pivet_value);
            identity_m[i] = identity_m[i].multiply_by_scalar(1.0 / pivet_value);
            for j in 0..m_copy.len() {
                if j != i {
                    let factor = m_copy[j].get(i);
                    m_copy[j] = m_copy[j].add_vector(&m_copy[i].multiply_by_scalar(-factor));
                    identity_m[j] =
                        identity_m[j].add_vector(&identity_m[i].multiply_by_scalar(-factor));
                }
            }
        }
        return Matrix::new(&identity_m);
    }

    fn swap_rows(&self, m: &mut Vec<RowVector>, row1: usize, row2: usize) {
        let temp = m[row1].clone();
        m[row1] = m[row2].clone();
        m[row2] = temp;
    }

    pub fn matrix(&self) -> &Vec<RowVector> {
        &self.matrix
    }

    pub fn get(&self, i: usize) -> &RowVector {
        &self.matrix[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::iter::zip;

    #[test]
    fn test_multiply_matrix() {
        let m1: Matrix = Matrix::new(&vec![
            RowVector::new(&vec![1.0, 2.0, 3.0]),
            RowVector::new(&vec![4.0, 5.0, 6.0]),
            RowVector::new(&vec![7.0, 8.0, 9.0]),
        ]);

        let m2: Matrix = Matrix::new(&vec![
            RowVector::new(&vec![9.0, 8.0, 7.0]),
            RowVector::new(&vec![6.0, 5.0, 4.0]),
            RowVector::new(&vec![3.0, 2.0, 1.0]),
        ]);

        let expected_m: Matrix = Matrix::new(&vec![
            RowVector::new(&vec![30.0, 24.0, 18.0]),
            RowVector::new(&vec![84.0, 69.0, 54.0]),
            RowVector::new(&vec![138.0, 114.0, 90.0]),
        ]);
        let output_m = m1.multiply(&m2);
        assert!(output_m == expected_m);
    }

    #[test]
    fn test_multiply_matrix_rectangular() {
        let m1: Matrix = Matrix::new(&vec![
            RowVector::new(&vec![1.0, 2.0]),
            RowVector::new(&vec![3.0, 4.0]),
            RowVector::new(&vec![5.0, 6.0]),
        ]);

        let m2: Matrix = Matrix::new(&vec![
            RowVector::new(&vec![7.0, 8.0, 9.0]),
            RowVector::new(&vec![10.0, 11.0, 12.0]),
        ]);

        let expected_m: Matrix = Matrix::new(&vec![
            RowVector::new(&vec![27.0, 30.0, 33.0]),
            RowVector::new(&vec![61.0, 68.0, 75.0]),
            RowVector::new(&vec![95.0, 106.0, 117.0]),
        ]);

        let output_m = m1.multiply(&m2);
        assert!(output_m == expected_m);
    }

    #[test]
    fn test_multiply_matrix_opposite_dimensions() {
        let m1: Matrix = Matrix::new(&vec![
            RowVector::new(&vec![1.0, 2.0, 3.0]),
            RowVector::new(&vec![4.0, 5.0, 6.0]),
        ]);

        let m2: Matrix = Matrix::new(&vec![
            RowVector::new(&vec![7.0, 8.0]),
            RowVector::new(&vec![9.0, 10.0]),
            RowVector::new(&vec![11.0, 12.0]),
        ]);

        let expected_m: Matrix = Matrix::new(&vec![
            RowVector::new(&vec![58.0, 64.0]),
            RowVector::new(&vec![139.0, 154.0]),
        ]);

        let output_m = m1.multiply(&m2);
        assert!(output_m == expected_m);
    }

    #[test]
    fn test_inverse_matrix() {
        let m: Matrix = Matrix::new(&vec![
            RowVector::new(&vec![1.0, 3.0, 5.0]),
            RowVector::new(&vec![2.0, 4.0, 6.0]),
            RowVector::new(&vec![2.0, 3.0, 1.0]),
        ]);
        let expected_m: Matrix = Matrix::new(&vec![
            RowVector::new(&vec![-7.0 / 3.0 as f32, 2.0, -1.0 / 3.0]),
            RowVector::new(&vec![5.0 / 3.0, -3.0 / 2.0, 2.0 / 3.0]),
            RowVector::new(&vec![-1.0 / 3.0, 1.0 / 2.0, -1.0 / 3.0]),
        ]);
        let inverse_m = m.inverse();
        assert!(
            zip(expected_m.matrix().iter(), inverse_m.matrix().iter()).all(|(v1, v2)| zip(
                v1.vector().iter(),
                v2.vector().iter()
            )
            .all(|(c1, c2)| (c1 - c2).abs() < 0.01))
        );
        let expected_identity_matrix = &inverse_m.multiply(&m);
        assert!(
            expected_identity_matrix
                .matrix()
                .iter()
                .enumerate()
                .all(|(i, v)| {
                    v.vector().iter().enumerate().all(|(j, c)| {
                        if i == j {
                            (c - 1.0).abs() < 0.01
                        } else {
                            (c - 0.0).abs() < 0.01
                        }
                    })
                })
        );
    }
    #[test]
    fn test_transpose_matrix() {
        let m: Matrix = Matrix::new(&vec![
            RowVector::new(&vec![1.0, 3.0, 5.0]),
            RowVector::new(&vec![2.0, 4.0, 6.0]),
            RowVector::new(&vec![2.0, 3.0, 1.0]),
            RowVector::new(&vec![2.0, 8.0, 1.0]),
        ]);
        let expected_m: Matrix = Matrix::new(&vec![
            RowVector::new(&vec![1.0, 2.0, 2.0, 2.0]),
            RowVector::new(&vec![3.0, 4.0, 3.0, 8.0]),
            RowVector::new(&vec![5.0, 6.0, 1.0, 1.0]),
        ]);
        let output_m = m.transpose();
        assert!(output_m == expected_m);
    }
}
