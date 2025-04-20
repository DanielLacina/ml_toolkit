use crate::dataframe::datatype::{DataType, DataTypeValue};
use std::{collections::{HashMap, HashSet}, hash::Hash};

const IDS: &str = "ids";

#[derive(Clone)]
pub struct DataFrame {
    index_to_column: HashMap<usize, String>,
    columns: HashMap<String, (usize, DataType, Vec<DataTypeValue>)>,
    len: usize,
}

impl DataFrame {
    pub fn new() -> Self {
        let mut df = Self {
            index_to_column: HashMap::new(),
            columns: HashMap::new(),
            len: 0,
        };
        df.insert_column(IDS, &Vec::new(), &DataType::Id);
        return df;
    }

    pub fn insert_column(
        &mut self,
        column_name: &str,
        values: &Vec<DataTypeValue>,
        dtype: &DataType,
    ) {
        // column values vector must have the same length as the other
        // column values vectors to keep everything consistent 
        assert!(values.len() == self.len());
        assert!(!(self.columns.contains_key(IDS) && column_name == IDS), "{}", format!("{} column cannot be modified", IDS));
        let header_index = self.columns.len();
        self.columns.insert(
            column_name.to_string(),
            (header_index, dtype.clone(), values.clone()),
        );
        self.index_to_column
            .insert(header_index, column_name.to_string());
    }

    fn update_ids(&mut self) {
        let len = self.len;
        let (_, values) = self.get_column_mut(IDS);
        for i in (values.len()..len) {
            values.push(DataTypeValue::Id(i));
        }
    }

    pub fn remove_column(&mut self, column_name: &str) {
        self.columns.remove(column_name);
    }

    fn get_column_mut(&mut self, column_name: &str) -> (&mut DataType, &mut Vec<DataTypeValue>) {
        let (_, dtype, values) = self.columns.get_mut(column_name).unwrap();
        return (dtype, values);
    }

    pub fn get_column(&self, column_name: &str) -> (&DataType, &Vec<DataTypeValue>) {
        let (_, dtype, values) = self.columns.get(column_name).unwrap();
        return (dtype, values);
    }

    pub fn get_column_by_index(&self, index: usize) -> (&String, &DataType, &Vec<DataTypeValue>) {
        let column_name = self.index_to_column.get(&index).unwrap();
        let (dtype, values) = self.get_column(column_name);
        return (column_name, dtype, values);
    }

    pub fn insert_row(&mut self, data_hashmap: &HashMap<String, DataTypeValue>) {
        assert!(data_hashmap.len() >= self.columns.len() - 1 && data_hashmap.len() <= self.columns.len());
        for (column_name, value) in data_hashmap.iter() {
            let (dtype, data) = self.get_column_mut(column_name);
            let right_dtype = match dtype {
                DataType::Float => {
                    matches!(value, DataTypeValue::Float(_))
                }
                DataType::Id => {
                    matches!(value, DataTypeValue::Id(_))
                }
                DataType::String => {
                    matches!(value, DataTypeValue::String(_))
                }
            };
            if !right_dtype {
                assert!(
                    matches!(value, DataTypeValue::Null),
                    "{}",
                    format!(
                        "value of {:?} is incompatible for column {} with dtype of {:?}",
                        value, column_name, dtype
                    )
                );
            }
            data.push(value.clone());
        }
        self.len += 1;
        self.update_ids();
    }

    pub fn convert_column_values_to_string(&mut self, column_name: &str) {
        let (_, dtype, data) = self.columns.get_mut(column_name).unwrap();
        for value in data.iter_mut() {
            match value {
                DataTypeValue::Float(inner) => {
                    *value = DataTypeValue::String(inner.to_string());
                },
                _ => {}
            }
        }
        *dtype = DataType::String;
    }

    pub fn columns(&self) -> Vec<&String> {
        (0..self.index_to_column.len())
            .map(|i| self.index_to_column.get(&i).unwrap())
            .collect()
    }

    pub fn get_columns_as_df(&self, columns: &Vec<String>) -> DataFrame {
        let mut df = DataFrame::new();
        df.len = self.len();
        for column_name in columns {
            let (dtype, data) = self.get_column(column_name);
            df.insert_column(column_name.as_str(), data, dtype);
        }
        df.update_ids();
        return df;
    }

    fn df_from_hashmap(&self, data_hashmap: &HashMap<String, (DataType, Vec<DataTypeValue>)>, len: usize) -> DataFrame {
        let mut df = DataFrame::new();
        df.len = len;
        for (column_name, (dtype, values)) in data_hashmap.iter() {
            df.insert_column(column_name, values, dtype);
        }  
        df.update_ids();
        return df;
    } 

    pub fn get_rows_as_df(&self, ids: Vec<usize>) -> DataFrame {
        let mut data_hashmap = HashMap::new();
        let column_names = self.columns(); 
        for column_name in column_names.iter() {
            let (dtype, values) = self.get_column(&column_name);
            let mut df_values = Vec::new(); 
            for id in ids.iter() {
                df_values.push(values[*id].clone());
            }
            data_hashmap.insert(column_name.clone().clone(),(dtype.clone(), df_values)); 
        }
        let df = self.df_from_hashmap(&data_hashmap, ids.len());
        return df;
    }

    pub fn bins(
        &self,
        column_name: &str,
        num_bins: usize,
    ) -> Vec<(DataTypeValue, DataTypeValue, u32)> {
        let (_, data) = self.get_column(column_name);
        let (_, ids) = self.get_column(IDS);
        let mut zipped: Vec<(DataTypeValue, DataTypeValue)> =
            ids.clone().into_iter().zip(data.clone()).collect();
        zipped.sort_by(|(_, a), (_, b)| a.cmp(b));
        let bin_size = zipped.len() / num_bins;
        let mut bins = Vec::new();
        for (i, (id, value)) in zipped.into_iter().enumerate() {
            bins.push((id, value, (i / bin_size) as u32));
        }
        bins.sort_by(|(a, _, _), (b, _, _)| a.cmp(b));
        return bins;
    }

    pub fn data(&self, include_ids: bool) -> HashMap<&String, (&DataType, &Vec<DataTypeValue>)> {
        let mut data_hashmap = HashMap::new();
        for (_, column_name) in self.index_to_column.iter() {
            if column_name == IDS && !include_ids {
                continue;
            }
            let (dtype, data) = self.get_column(column_name);
            data_hashmap.insert(column_name, (dtype, data));
        }
        return data_hashmap;
    }

    pub fn median(&self, column_name: &str) -> f32 {
        let (_, values) = self.get_column(column_name);
        let mut values: Vec<f32> = values
            .iter()
            .filter_map(|value| match value {
                DataTypeValue::Float(inner) => Some(*inner),
                _ => None,
            })
            .collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if values.len() % 2 == 0 {
            return (values[values.len() / 2] + values[(values.len() - 1) / 2]) / 2.0;
        } else {
            return values[values.len() / 2];
        }
    }

    pub fn mean(&self, column_name: &str) -> f32 {
        let mut sum = 0.0;
        let (dtype, values) = self.get_column(column_name);
        if !matches!(dtype, DataType::Float) {
            panic!("the mean can only be found from float values");
        }
        for value in values {
            match value {
                DataTypeValue::Float(inner) => {
                    sum += *inner;
                }
                _ => {}
            }
        }
        let mean = sum / values.len() as f32;
        return mean;
    }

    pub fn std(&self, column_name: &str, mean: Option<f32>) -> f32 {
        let mean = if let Some(mean) = mean {
            mean
        } else {
            self.mean(column_name)
        };
        let mut sum = 0.0;
        let (dtype, values) = self.get_column(column_name);
        if !matches!(dtype, DataType::Float) {
            panic!("the mean can only be found from float values");
        }
        for value in values {
            match value {
                DataTypeValue::Float(inner) => {
                    sum += (*inner - mean).powf(2.0);
                }
                _ => {}
            }
        }
        let std = f32::sqrt(sum / (values.len() - 1) as f32);
        return std;
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::csv::df_from_csv;

    #[test]
    fn test_df() {
        let filename = "housing.csv";
        let row_limit = 1000;
        let df = df_from_csv(filename, Some(row_limit));
        let column_names = df.columns();
        assert!(df.len() == row_limit);
        assert!(
            column_names
                == vec![
                    IDS,
                    "longitude",
                    "latitude",
                    "housing_median_age",
                    "total_rooms",
                    "total_bedrooms",
                    "population",
                    "households",
                    "median_income",
                    "median_house_value",
                    "ocean_proximity"
                ]
        );

        assert!(
           !column_names 
                .iter().any(|column_name| {
                   let (_, data) = df.get_column(column_name);
                 data.iter()
                .all(|value| matches!(value, DataTypeValue::Null))
                }) 
        ) ;
        assert!(
           column_names 
                .iter()
                .all(|column_name| 
                {    
                    let (_, data) = df.get_column(column_name);
                    data.len() == row_limit })
        );
        assert!(column_names.iter().all(|column_name| {
            let (dtype, data) = df.get_column(column_name);
            match dtype {
                DataType::Float => data.iter().all(|value| match value {
                    DataTypeValue::Float(_) => true,
                    DataTypeValue::Null => true,
                    _ => false,
                }),
                DataType::String => data.iter().all(|value| match value {
                    DataTypeValue::String(_) => true,
                    DataTypeValue::Null => true,
                    _ => false,
                }),
                _ => true,
            }
        }));
         assert!(column_names.iter().all(|column_name| {
            let (dtype, _) = df.get_column(column_name);
            if column_name.as_str() == "ocean_proximity" {
                matches!(dtype, DataType::String)
            } else if column_name.as_str() == IDS {
                matches!(dtype, DataType::Id)   
            }  else {
                matches!(dtype, DataType::Float)
            }
        }));
    }

    #[test]
    fn test_get_columns_as_df() {
        let filename = "housing.csv";
        let row_limit = 10;
        let df = df_from_csv(filename, Some(row_limit));
        let columns = vec!["median_income".to_string(), "households".to_string(), "total_bedrooms".to_string()]; 
        let new_df = df.get_columns_as_df(&columns);   
        assert!(new_df.columns().len() == columns.len() + 1);
        assert!(columns.iter().all(|column_name| {
            let (_, values) = new_df.get_column(column_name);
            values.len() == row_limit
        }));
        let (_, values) = new_df.get_column(IDS); 
        assert!(values.len() == row_limit);
    }

    #[test]
    fn test_get_data() {
        let filename = "housing.csv";
        let row_limit = 10;
        let df = df_from_csv(filename, Some(row_limit));
        let data = df.data(true);
        let column_names = vec![
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
            "median_house_value",
            "ocean_proximity",
        ];
        assert!(
            column_names
                .iter()
                .all(|column_name| data.contains_key(&column_name.to_string()))
        );
        assert!(column_names.iter().all(|column_name| {
            let (_, data) = df.get_column(column_name);
            data.len() == row_limit
        }));
    }

    #[test]
    #[should_panic]
    fn test_insert_column_invalid_values_length() {
        let mut df = DataFrame::new();
        let column_name = "ocean_proximity";
        let values = vec!["near ocean", "far from ocean"].iter().map(|value| DataTypeValue::String(value.to_string())).collect();
        let dtype = DataType::String;
        df.insert_column(column_name, &values, &dtype);
     }

     #[test]
     fn test_insert_column() {
        let mut df = DataFrame::new();
        let column_name = "ocean_proximity";
        let dtype = DataType::String;
        df.insert_column(column_name, &Vec::new(), &dtype);
        let (column_dtype, column_values) = df.get_column(column_name); 
        assert!(matches!(column_dtype, dtype));
        assert!(column_values.len() == 0);
     }

     #[test]
     fn test_get_columns() {
        let filename = "housing.csv";
        let row_limit = 10;
        let df = df_from_csv(filename, Some(row_limit));
        let (column_dtype, column_values) = df.get_column("ocean_proximity");  
        assert!(column_values.len() == row_limit);
        assert!(matches!(column_dtype, DataType::String));
     }

     #[test]
     fn test_convert_column_values_to_string() {
        let filename = "housing.csv";
        let row_limit = 10;
        let mut df = df_from_csv(filename, Some(row_limit));
        let column_name = "median_income";
        df.convert_column_values_to_string(column_name);        
        let (dtype, values) = df.get_column(column_name);
        assert!(matches!(dtype, DataType::String));
        assert!(values.iter().all(|value| {
             matches!(value, DataTypeValue::String(_)) || matches!(value, DataTypeValue::Null) 
        }));
     }

     #[test]
     #[should_panic]
     fn test_insert_row_invalid_column() {
        let mut df = DataFrame::new();
        df.insert_column("income", &Vec::new(), &DataType::Float);
        let mut data_hashmap = HashMap::new();
        data_hashmap.insert("not income".to_string(), DataTypeValue::Float(10.0)); 
        df.insert_row(&data_hashmap);
     }

     #[test]
     #[should_panic]
     fn test_insert_row_not_specifying_all_columns() {
        let mut df = DataFrame::new();
        df.insert_column("income", &Vec::new(), &DataType::Float);
        df.insert_column("location", &Vec::new(), &DataType::String);
        let mut data_hashmap = HashMap::new();
        data_hashmap.insert("income".to_string(), DataTypeValue::Float(10.0)); 
        df.insert_row(&data_hashmap);
     }

     #[test]
     fn test_insert_row() {
        let mut df = DataFrame::new();
        df.insert_column("income", &Vec::new(), &DataType::Float);
        df.insert_column("location", &Vec::new(), &DataType::String);
        let mut data_hashmap = HashMap::new();
        let income = DataTypeValue::Float(10.0); 
        let location = DataTypeValue::String("Washington".to_string());
        data_hashmap.insert("income".to_string(), income.clone()); 
        data_hashmap.insert("location".to_string(), location.clone());
        df.insert_row(&data_hashmap);
        assert!(data_hashmap.iter().all(|(column_name, value)| {
            let (_, values) = df.get_column(column_name);  
            *values == vec![value.clone()] 
        }));
     }

     #[test]
     fn test_get_columns_by_index() {
        let filename = "housing.csv";
        let row_limit = 10;
        let df = df_from_csv(filename, Some(row_limit));
        // columns ordered by index 
        let columns = df.columns(); 
        let (dtype, _) = df.get_column(columns[0]); 
        let (column_name, column_dtype, column_values) = df.get_column_by_index(0);  
        assert!(column_values.len() == row_limit);
        assert!(matches!(column_dtype, dtype));
        assert!(column_name == columns[0]);
     }


    #[test]
    fn test_get_mean() {
        let filename = "housing.csv";
        let row_limit = 10;
        let df = df_from_csv(filename, Some(row_limit));
        let mut means: HashMap<String, f32> = HashMap::new();
        means.insert("longitude".to_string(), -122.24500);
        means.insert("latitude".to_string(), 37.85000);
        means.insert("housing_median_age".to_string(), 46.80000);
        means.insert("total_rooms".to_string(), 2500.90000);
        means.insert("total_bedrooms".to_string(), 470.10000);
        means.insert("population".to_string(), 976.30000);
        means.insert("households".to_string(), 458.20000);
        means.insert("median_income".to_string(), 4.99608);
        means.insert("median_house_value".to_string(), 314480.00000);
        assert!(
            means
                .iter()
                .all(|(column_name, mean)| (df.mean(column_name) - mean).abs() < 0.01)
        );
    }

    #[test]
    fn test_get_std() {
        let filename = "housing.csv";
        let row_limit = 10;
        let df = df_from_csv(filename, Some(row_limit));

        let mut stds: HashMap<String, f32> = HashMap::new();
        stds.insert("longitude".to_string(), 0.011785);
        stds.insert("latitude".to_string(), 0.012472);
        stds.insert("housing_median_age".to_string(), 10.064238);
        stds.insert("total_rooms".to_string(), 1858.210456);
        stds.insert("total_bedrooms".to_string(), 315.910483);
        stds.insert("population".to_string(), 648.036702);
        stds.insert("households".to_string(), 323.469662);
        stds.insert("median_income".to_string(), 2.243359);
        stds.insert("median_house_value".to_string(), 68355.310287);
        assert!(stds.iter().all(|(column_name, std)| {
            let df_std = df.std(column_name, None);
            (df_std - std).abs() < 0.01
        }));
    }

    #[test]
    fn test_get_median() {
        let filename = "housing.csv";
        let row_limit = 10;
        let df = df_from_csv(filename, Some(row_limit));
        let mut medians: HashMap<String, f32> = HashMap::new();
        medians.insert("longitude".to_string(), -122.2500);
        medians.insert("latitude".to_string(), 37.8500);
        medians.insert("housing_median_age".to_string(), 52.0000);
        medians.insert("total_rooms".to_string(), 2081.0000);
        medians.insert("total_bedrooms".to_string(), 384.5000);
        medians.insert("population".to_string(), 829.5000);
        medians.insert("households".to_string(), 386.5000);
        medians.insert("median_income".to_string(), 3.9415);
        medians.insert("median_house_value".to_string(), 320250.0000);
        assert!(medians.iter().all(|(column_name, median)| {
            let df_median = df.median(column_name);
            (df_median - median).abs() < 0.01
        }));
    }

    #[test]
    fn test_get_bins() {
        let filename = "housing.csv";
        let row_limit = 100;
        let bins = 5;
        let bin_size = row_limit/bins; 
        let df = df_from_csv(filename, Some(row_limit));
        let mut bins = df.bins("population", bins);
        bins.sort_by(|(_, _, a), (_, _, b)| a.cmp(b)); 
        assert!(bins.iter().enumerate().all(|(i, (_, a_value, bin_value))| {
             let bin_row_count_by_bin = i/(*bin_value as usize + 1); 
             let bin_cmp_start = (i as i32 - bin_row_count_by_bin as i32 - bin_size as i32).max(0) as usize;
             let bin_cmp_end = i - bin_row_count_by_bin; 
             for (_, b_value, _) in bins[bin_cmp_start..bin_cmp_end].iter() {
                if !(a_value >= b_value) {
                    return false;
                }
             }  
             true 
        }));
    }
}
