use crate::dataframe::{DataFrame, DataType, DataTypeValue};

pub enum ImputerStrategy {
    Median,
}

pub enum Scalar {
    None,
    Standard,
}

pub struct Pipeline {
    imputer_strategy: ImputerStrategy,
    scalar: Scalar,
}

impl Pipeline {
    pub fn new(
        imputer_strategy: ImputerStrategy,
        scalar: Scalar,
    ) -> Self {
        Self {
            imputer_strategy,
            scalar,
        }
    }

    pub fn transform(&self, df: &DataFrame) -> Vec<Vec<f32>> {
        let mut output_matrix = vec![vec![]; df.len()];
        let column_names = df.columns(); 
        let data = df.data(false);
        for (j, column_name) in column_names.iter().enumerate() {
            if column_name.as_str() == "ids" {
                continue;
            }
            let (dtype, values) = data.get(column_name).unwrap();   
            match dtype {
                DataType::Float => {
                    let median = df.median(column_name);
                    for (i, value) in values.iter().enumerate() {
                        match value {
                            DataTypeValue::Float(inner) => {
                                output_matrix[i].push(*inner);
                            }
                            DataTypeValue::Null => match self.imputer_strategy {
                                ImputerStrategy::Median => {
                                    output_matrix[i].push(median);
                                }
                            },
                            _ => panic!("value type is inconsistent with column datatype header"),
                        }
                    }
                }
                DataType::String => panic!("string data must be encoded"),
                _ => {
                    panic!("only columns with column datatype header of float can be processed")
                }
            }
        }
        self.scale_data(&mut output_matrix);
        return output_matrix;
    }


    fn scale_data(&self, matrix: &mut Vec<Vec<f32>>) {
        match self.scalar {
            Scalar::Standard => {
                for i in (0..matrix[0].len()) {
                    let column_vector = matrix.iter().map(|v| v[i]).collect();
                    let mean = self.mean(&column_vector);
                    let std = self.std(&column_vector, Some(mean));
                    for row_vector in matrix.iter_mut() {
                        row_vector[i] = (row_vector[i] - mean) / std;
                    }
                }
            }
            Scalar::None => {}
        }
    }

    fn mean(&self, column_vector: &Vec<f32>) -> f32 {
        let mut sum = 0.0;
        for value in column_vector {
            sum += *value;
        }
        return sum / column_vector.len() as f32;
    }

    fn std(&self, column_vector: &Vec<f32>, mean: Option<f32>) -> f32 {
        let mean = if let Some(mean) = mean {
            mean
        } else {
            self.mean(column_vector)
        };
        let mut sum = 0.0;
        for value in column_vector {
            sum += (value - mean).powf(2.0);
        }
        let std = f32::sqrt(sum / (column_vector.len() - 1) as f32);
        return std;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::csv::df_from_csv;

    #[test]
    fn test_pipeline_transform() {
        let imputer_strategy = ImputerStrategy::Median;
        let scalar = Scalar::Standard;
        let pipeline = Pipeline::new( imputer_strategy, scalar);
        let df = df_from_csv("housing.csv", Some(10000));
        let output_matrix = pipeline.transform(&df);
        assert!(output_matrix.iter().all(|v| { v.len() == 14 }));
    }
}
