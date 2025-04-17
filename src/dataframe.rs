use std::collections::{HashMap, HashSet};
use std::iter::zip;
use std::fs::File;
use std::io::{self, BufReader, prelude::*};
use std::cmp::Ordering;

const IDS: &str = "ids";

#[derive(Clone, Debug)]
pub enum DataTypeValue {
    Null,
    Float(f32),
    String(String),
}

impl PartialOrd for DataTypeValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    } 
}

impl Ord for DataTypeValue {
    fn cmp(&self, other: &Self) -> Ordering {
        let values: Vec<Option<f32>> = vec![self, other].iter().map(|dtype| {
            match dtype {
            Self::Float(inner) => {
                Some(*inner)
            },  
            Self::Null => {
                None
            },
            _ => panic!("other datatypes cannot be compared")
        }}).collect();
        let current = values[0];
        let other = values[1];
        if current.is_none() && other.is_none() {
            return Ordering::Equal;
        } 
        else if current.is_none() {
            return Ordering::Less;
        }
        else if other.is_none() {
            return Ordering::Greater;
        }
        let current = current.unwrap(); 
        let other = other.unwrap();
        if (current - other).abs() < 0.1 {
            return Ordering::Equal;
        } 
        else if current > other {
            return Ordering::Greater;
        } 
        else {
            return Ordering::Less;
        }
    }
} 

impl PartialEq for DataTypeValue {
    fn eq(&self, other: &Self) -> bool {
        let values: Vec<Option<f32>> = vec![self, other].iter().map(|dtype| {
            match dtype {
            Self::Float(inner) => {
                Some(*inner)
            },  
            Self::Null => {
                None
            },
            _ => panic!("other datatypes cannot be compared")
        }}).collect();
        let current = values[0];
        let other = values[1];
        if current.is_none() && other.is_none() {
            return true;
        } 
        else if current.is_none() || other.is_none() {
            return false;
        }
        let current = current.unwrap(); 
        let other = other.unwrap();
        if (current - other).abs() < 0.1 {
            return true;
        } else {
            return false;
        } 
    }
}

impl Eq for DataTypeValue {}

#[derive(Clone, Debug)]
pub enum DataType {
    Float,
    String,
}


#[derive(Clone)]
pub struct DataFrame {
    index_to_column: HashMap<usize, String>,
    columns: HashMap<String, (usize, DataType, Vec<DataTypeValue>)>,
    len: usize,
}

impl DataFrame {
    pub fn new() -> Self {
        return Self {
            index_to_column: HashMap::new(),
            columns: HashMap::new(),
            len: 0,
        };
    }

    // pub fn from_hashmap(hashmap: &HashMap<String, Vec<f32>>) -> Self {
    //     let mut df = DataFrame::new();
    //     let mut len = None; 
    //     for (column_name, values) in hashmap.iter() {
    //         df.insert_header(column_name); 
    //         let (_, _, data) = df.columns.get_mut(column_name).unwrap();
    //         *data = values.iter().map(|value| DataTypeValue::Float(*value)).collect();
    //         if let Some(len) = len {
    //             assert!(data.len() == len);
    //         } else {
    //             len = Some(data.len()); 
    //         }
    //     }
    //     df.len = len.unwrap();
    //     return df;
    // }
    

    pub fn from_csv(filename: &str, row_limit: Option<usize>) -> Self {
        let mut df = DataFrame::new();
        let file = File::open(filename).unwrap();
        let reader = BufReader::new(file);
        let mut reader_lines = reader.lines();
        let headers = reader_lines.next().unwrap().unwrap();
        df.insert_headers_as_string(&headers);
        let row_limit = if let Some(limit) = row_limit {
            limit
        } else {
            1000000
        };
        for (i, line) in reader_lines.enumerate() {
            if i == row_limit {
                break;
            }
            let line = line.unwrap();
            df.insert_row_as_string(&line);
        }
        return df;
    }



    fn insert_headers_as_string(&mut self, headers: &str) {
        for header in headers.split(",") {
            self.insert_header(header);
        }
    }

    fn insert_header(&mut self, header: &str) {
        let header_index = self.columns.len();
        self.columns.insert(
            header.to_string(),
            (header_index, DataType::Float, Vec::new()),
        );
        self.index_to_column
            .insert(header_index, header.to_string());
    }

    fn insert_ids(&mut self, count: usize) {
        let ids = (0..count).map(|i| DataTypeValue::Float(i as f32)).collect(); 
        self.insert_column(IDS, &ids);
    }

    fn insert_row_as_string(&mut self, line: &str) {
        for (i, value) in line.split(",").enumerate() {
            let column_name = self.index_to_column.get(&i).unwrap().clone();
            let (dtype, _) = self.get_column(&column_name);
            if value == "" {
                self.insert_value(&column_name, DataTypeValue::Null);
            } else {
                match dtype {
                    DataType::Float => match value.parse::<f32>() {
                        Ok(parsed_value) => {
                            self.insert_value(&column_name, DataTypeValue::Float(parsed_value));
                        }
                        Err(_) => {
                            let column_name = self.index_to_column.get(&i).unwrap().clone();
                            self.convert_column_values_to_string(column_name.clone().as_str());
                            self.insert_value(
                                &column_name,
                                DataTypeValue::String(value.to_string()),
                            );
                        }
                    },
                    DataType::String => {
                        self.insert_value(&column_name, DataTypeValue::String(value.to_string()));
                    }
                }
            }
        }
        self.len += 1;
        self.insert_ids(self.len);
    }

    fn insert_value(&mut self, column_name: &str, value: DataTypeValue) {
        let (_, data) = self.get_column_mut(column_name);
        data.push(value);
    }

    fn convert_column_values_to_string(&mut self, column_name: &str) {
        let (dtype, data) = self.get_column_mut(column_name);
        for value in data.iter_mut() {
            match value {
                DataTypeValue::Float(inner) => {
                    *value = DataTypeValue::String(inner.to_string());
                }
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

    pub fn remove_column(&mut self, column_name: &str) {
        self.columns.remove(column_name); 
    } 

    fn insert_column(&mut self, column_name: &str, values: &Vec<DataTypeValue>) {
        if self.columns.len() == 0 {
            self.len = values.len(); 
            self.insert_ids(values.len());
        }
        self.insert_header(column_name);
        let (_, data) = self.get_column_mut(column_name);
        *data = values.clone(); 
    }

    pub fn get_columns_as_df(&self, columns: &Vec<String>) -> DataFrame {
        let mut df = DataFrame::new(); 
        for column_name in columns {
            let (_, _, data) = self.columns.get(column_name.as_str()).unwrap();
            df.insert_column(column_name.as_str(), data);
        }
        return df;
    }

    pub fn bins(&self, column_name: &str, num_bins: usize) -> Vec<i32> {
        let (_, data)  = self.get_column(column_name);
        let (_, ids) = self.get_column(IDS);
        let mut data = data.clone();
        let mut ids = ids.clone();
    }

    // pub fn remove_columns(&mut self, except: HashSet<String>) {
    //     let columns_to_remove: Vec<String> = self.columns.iter().filter_map(|(column_name, (_, _, _))| {
    //         if except.contains(column_name.as_str()) {
    //             return None;
    //         } else {
    //             return Some(column_name.clone());
    //         }
    //     }).collect();
    //     for column in columns_to_remove {
    //         self.columns.remove(&column);
    //     }
    // }

    pub fn data(&self) -> HashMap<&String, (&DataType, &Vec<DataTypeValue>)> {
        let mut data_hashmap = HashMap::new();
        for (_, column_name) in self.index_to_column.iter() {
           let (dtype, data) = self.get_column(column_name);
           data_hashmap.insert(column_name, (dtype, data));
        }
        return data_hashmap;
    }

    pub fn get_column(&self, column_name: &str) -> (&DataType, &Vec<DataTypeValue>) {
        let (_, dtype, values) = self.columns.get(column_name).unwrap();
        return (dtype, values);
    }

    fn get_column_mut(&mut self, column_name: &str) -> (&mut DataType, &mut Vec<DataTypeValue>) {
        let (_, dtype, values) = self.columns.get_mut(column_name).unwrap();
        return (dtype, values);
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

    #[test]
    fn test_read_from_csv() {
        let filename = "housing.csv";
        let row_limit = 1000;
        let df = DataFrame::from_csv(filename, Some(row_limit));
        let column_names = df.columns();
        assert!(df.len() == row_limit);
        assert!(
            column_names
                == vec![
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
        assert!(!df.columns.iter().any(|(_, (_, _, data))| {
            data.iter()
                .all(|value| matches!(value, DataTypeValue::Null))
        }));
        assert!(
            df.columns
                .iter()
                .all(|(_, (_, _, data))| { data.len() == row_limit })
        );
        assert!(df.columns.iter().all(|(_, (_, dtype, data))| {
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
            }
        }));
        assert!(df.columns.iter().all(|(column_name, (_, dtype, _))| {
            if column_name == "ocean_proximity" {
                matches!(dtype, DataType::String)
            } else {
                matches!(dtype, DataType::Float)
            }
        }));
    }

    #[test]
    fn test_get_data() {
        let filename = "housing.csv";
        let row_limit = 10;
        let df = DataFrame::from_csv(filename, Some(row_limit));
        let data = df.data();
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
        assert!(
            df.columns
                .iter()
                .all(|(_, (_, _, data))| { data.len() == row_limit })
        );
    }

    #[test]
    fn test_get_mean() {
        let filename = "housing.csv";
        let row_limit = 10;
        let df = DataFrame::from_csv(filename, Some(row_limit));
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
        let df = DataFrame::from_csv(filename, Some(row_limit));

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
        let df = DataFrame::from_csv(filename, Some(row_limit));
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
}
