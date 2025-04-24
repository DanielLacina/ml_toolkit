use crate::{
    dataframe::{DataFrame, DataTypeValue},
    pipeline::transformers::Transformer,
};

pub struct StandardScalar;

impl StandardScalar {
    pub fn new() -> Self {
        Self
    }
}
impl Transformer for StandardScalar {
    fn transform(&self, df: &DataFrame, column_names: &Vec<String>) -> DataFrame {
        let mut df = df.clone();
        let df_column_names: Vec<String> = df
            .columns()
            .into_iter()
            .filter(|df_column_name| {
                *df_column_name != DataFrame::id_column() && column_names.contains(*df_column_name)
            })
            .map(|column_name| column_name.clone())
            .collect();
        for df_column_name in df_column_names {
            let mean = df.mean(&df_column_name);
            let std = df.std(&df_column_name, Some(mean));
            for i in (0..df.len()) {
                let current_value = df.get_cell_value(&df_column_name, i).clone();
                let current_value = match current_value {
                    DataTypeValue::Float(inner) => inner,
                    _ => panic!(
                        "invalid datatype: {:?} for standard scalar in column {}",
                        current_value, df_column_name
                    ),
                };
                let new_value = (current_value - mean) / std;
                df.modify_cell(&df_column_name, i, DataTypeValue::Float(new_value));
            }
        }
        return df;
    }
}

#[cfg(test)]
mod tests {
    use crate::{dataframe::{csv::df_from_csv, DataTypeValue}, pipeline::transformers::Transformer};
    use std::iter::zip;

    use super::StandardScalar;
    #[test]
    #[should_panic]
   fn test_std_scalar_with_strings() {
      let df = df_from_csv("housing.csv", Some(100));
      let std_scalar = StandardScalar::new();
      let columns = df.columns().into_iter().map(|column| column.clone()).collect();
      std_scalar.transform(&df, &columns);
   }
    #[test]
   fn test_std_scalar() {
      let df = df_from_csv("housing.csv", Some(100));
      let numeric_columns: Vec<String> = df.numeric_columns().iter().map(|column| column.clone().clone()).collect();
      let std_scalar = StandardScalar::new();
      let df_scaled = std_scalar.transform(&df, &numeric_columns);
      assert!(numeric_columns.iter().all(|column| {
           let mean = df.mean(column);
           let std = df.std(column, Some(mean));
           let (_, values) = df.get_column(column); 
           let (_, scaled_values) = df_scaled.get_column(column); 
           for (value, scaled_value) in zip(values, scaled_values) {
              let value = match value {
                 DataTypeValue::Float(inner) => {
                    inner
                 }, 
                 _ => panic!("value must be float")
              };
              let expected_scaled_value = DataTypeValue::Float((value - mean)/std);
              if expected_scaled_value != *scaled_value {
                return false
              } 
           }
           return true;
        }
      ));
   }
}
