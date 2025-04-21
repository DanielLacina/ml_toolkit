use crate::{dataframe::{DataFrame, DataTypeValue}, pipeline::scalars::scalar::Scalar}; 

pub struct StandardScalar; 

impl StandardScalar {
    pub fn new() -> Self {
         Self
    }
}
impl Scalar for StandardScalar {
    fn scale_data(&self, df: &DataFrame, column_names: &Vec<String>) -> DataFrame {
        let mut df = df.clone();
        let df_column_names: Vec<String> = df.columns().into_iter().filter(|df_column_name| *df_column_name != DataFrame::id_column() && column_names.contains(*df_column_name)).map(|column_name| column_name.clone()).collect();
        for df_column_name in df_column_names {
            let mean = df.mean(&df_column_name);
            let std = df.std(&df_column_name, Some(mean)); 
            for i in (0..df.len()) {
                let current_value = df.get_cell_value(&df_column_name, i).clone(); 
                let current_value = match current_value {
                      DataTypeValue::Float(inner) => {
                         inner
                      },
                      _ => panic!("invalid datatype: {:?} for standard scalar in column {}", current_value, df_column_name), 
                };
                let new_value = (current_value - mean)/std;
                df.modify_cell(&df_column_name, i, DataTypeValue::Float(new_value));
            }
        }
        return df;
    }
} 