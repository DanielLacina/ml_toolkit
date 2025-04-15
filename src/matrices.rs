use std::iter::zip;



pub fn transpose_matrix(m: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut output_matrix = Vec::new();
    for i in 0..m[0].len() {
        let mut output_vector = Vec::new();
        for j in 0..m.len() {
            output_vector.push(m[j][i]); 
        }
        output_matrix.push(output_vector);
    }
    return output_matrix;
}

pub fn multiply_matrices(m1: &Vec<Vec<f32>>, m2: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut output_matrix = Vec::new();
    for i in 0..m1.len() {
        let mut new_row = Vec::new();
        for j in 0..m2[0].len() {
            let mut sum = 0.0;
            for k in 0..m2.len() {
                sum += m1[i][k] * m2[k][j];
            } 
            new_row.push(sum);
        }   
        output_matrix.push(new_row);
     }
     return output_matrix;
}

pub fn identity_matrix(size: usize) -> Vec<Vec<f32>> {
    let mut matrix = Vec::new();
    for i in 0..size {
        let mut vector = Vec::new();
        for j in 0..size {
            if i == j {
                vector.push(1.0);
            } else {
                vector.push(0.0)
            } 
        } 
        matrix.push(vector);
    }
    return matrix;
}

pub fn inverse_matrix(m: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
   let mut m_copy = m.clone(); 
   let mut identity_m = identity_matrix(m.len());
   for i in 0..m_copy.len() {
       let mut pivet_row_i = i; 
       while pivet_row_i < m_copy.len() && m_copy[pivet_row_i][i] == 0.0 {
          pivet_row_i += 1;
       } 
       if pivet_row_i == m_copy.len() {
        panic!("cannot derive the identity matrix with a column of all zeros");
       }   
       if pivet_row_i != i {
          swap_rows(&mut m_copy, pivet_row_i, i);
          swap_rows(&mut identity_m, pivet_row_i, i);
          // i now indexes the previous row indexed by pivet_row_i and vice versa
       }
       let pivet_value = m_copy[i][i]; 
       // sets the matrix[row_index][row_index] to 1 
       m_copy[i] = multiply_vector_by_scalar(&m_copy[i], 1.0/pivet_value);
       identity_m[i] = multiply_vector_by_scalar(&identity_m[i], 1.0/pivet_value);
       for j in 0..m_copy.len() {
          if j != i {
            let factor = m_copy[j][i];
            m_copy[j] = add_vectors(&m_copy[j], &multiply_vector_by_scalar(&m_copy[i], -factor));  
            identity_m[j] = add_vectors(&identity_m[j], &multiply_vector_by_scalar(&identity_m[i], -factor));  
          }
       }   
   } 
   return identity_m;
}

fn swap_rows(m: &mut Vec<Vec<f32>>, row1: usize, row2: usize) {
    let temp = m[row1].clone();   
    m[row1] = m[row2].clone();
    m[row2] = temp;
}

fn multiply_vector_by_scalar(v: &Vec<f32>, scalar: f32) -> Vec<f32> {
    let v = v.clone(); 
    let scaled_vector = v.into_iter().map(|c| c * scalar).collect();
    return scaled_vector;
}

fn add_vectors(v1: &Vec<f32>, v2: &Vec<f32>) -> Vec<f32> {
    let summed_vectors = zip(v1, v2).map(|(c1, c2)| *c1 + *c2).collect();
    return summed_vectors;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiply_matrix() {
        let m1: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let m2: Vec<Vec<f32>> = vec![
            vec![9.0, 8.0, 7.0],
            vec![6.0, 5.0, 4.0],
            vec![3.0, 2.0, 1.0],
        ];

        let expected_m:  Vec<Vec<f32>> = vec![
            vec![30.0, 24.0, 18.0],
            vec![84.0, 69.0, 54.0],
            vec![138.0, 114.0, 90.0],
        ]; 
        let output_m = multiply_matrices(&m1, &m2); 
        assert!(output_m == expected_m);
    } 

    #[test]
fn test_multiply_matrix_rectangular() {
    let m1: Vec<Vec<f32>> = vec![
        vec![1.0, 2.0],  
        vec![3.0, 4.0],
        vec![5.0, 6.0], 
    ];

    let m2: Vec<Vec<f32>> = vec![
        vec![7.0, 8.0, 9.0],   
        vec![10.0, 11.0, 12.0] 
    ];

    let expected_m: Vec<Vec<f32>> = vec![
        vec![27.0, 30.0, 33.0],
        vec![61.0, 68.0, 75.0],
        vec![95.0, 106.0, 117.0],
    ];

    let output_m = multiply_matrices(&m1, &m2);
    assert_eq!(output_m, expected_m);
    }

    #[test]
fn test_multiply_matrix_opposite_dimensions() {
    let m1: Vec<Vec<f32>> = vec![
        vec![1.0, 2.0, 3.0],  
        vec![4.0, 5.0, 6.0],  
    ];

    let m2: Vec<Vec<f32>> = vec![
        vec![7.0, 8.0],       
        vec![9.0, 10.0],
        vec![11.0, 12.0],   
    ];

    let expected_m: Vec<Vec<f32>> = vec![
        vec![58.0, 64.0],
        vec![139.0, 154.0],
    ];

    let output_m = multiply_matrices(&m1, &m2);
    assert_eq!(output_m, expected_m);
   }

    #[test]
    fn test_inverse_matrix() {
        let m: Vec<Vec<f32>> = vec![
            vec![1.0, 3.0, 5.0],
            vec![2.0, 4.0, 6.0],
            vec![2.0, 3.0, 1.0],
        ];
        let expected_m: Vec<Vec<f32>> = vec![
            vec![-7.0/3.0 as f32, 2.0, -1.0/3.0],
            vec![5.0/3.0, -3.0/2.0, 2.0/3.0],
            vec![-1.0/3.0, 1.0/2.0, -1.0/3.0],
        ];
        let inverse_m = inverse_matrix(&m);
        assert!(zip(expected_m.iter(), inverse_m.iter()).all(|(v1, v2)| zip(v1, v2).all(|(c1, c2)| (c1 - c2).abs() < 0.01)));
        let expected_identity_matrix = multiply_matrices(&inverse_m, &m);
        assert!(expected_identity_matrix.iter().enumerate().all(|(i, v)| v.iter().enumerate().all(|(j, c)| {
             if i == j {
                (c - 1.0).abs() < 0.01 
             } else {
                (c - 0.0).abs() < 0.01
             }    
        })));
    }
    #[test]
    fn test_transpose_matrix() {
        let m: Vec<Vec<f32>> = vec![
            vec![1.0, 3.0, 5.0],
            vec![2.0, 4.0, 6.0],
            vec![2.0, 3.0, 1.0],
            vec![2.0, 8.0, 1.0],
        ];
        let expected_m: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 2.0, 2.0],
            vec![3.0, 4.0, 3.0, 8.0],
            vec![5.0, 6.0, 1.0, 1.0],
        ];
        let output_m = transpose_matrix(&m);
        assert!(output_m == expected_m);
    }

}
