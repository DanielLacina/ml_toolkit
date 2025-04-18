use std::cmp::Ordering;

#[derive(Clone, Debug)]
pub enum DataTypeValue {
    Null,
    Float(f32),
    String(String),
    Id(usize),
}

impl PartialOrd for DataTypeValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DataTypeValue {
    fn cmp(&self, other: &Self) -> Ordering {
        let values: Vec<Option<f32>> = vec![self, other]
            .iter()
            .map(|dtype| match dtype {
                Self::Float(inner) => Some(*inner),
                Self::Id(inner) => Some(*inner as f32),
                Self::Null => None,
                _ => panic!("other datatypes cannot be compared"),
            })
            .collect();
        let current = values[0];
        let other = values[1];
        if current.is_none() && other.is_none() {
            return Ordering::Equal;
        } else if current.is_none() {
            return Ordering::Less;
        } else if other.is_none() {
            return Ordering::Greater;
        }
        let current = current.unwrap();
        let other = other.unwrap();
        if (current - other).abs() < 0.1 {
            return Ordering::Equal;
        } else if current > other {
            return Ordering::Greater;
        } else {
            return Ordering::Less;
        }
    }
}

impl PartialEq for DataTypeValue {
    fn eq(&self, other: &Self) -> bool {
        let values: Vec<Option<f32>> = vec![self, other]
            .iter()
            .map(|dtype| match dtype {
                Self::Float(inner) => Some(*inner),
                Self::Id(inner) => Some(*inner as f32),
                Self::Null => None,
                _ => panic!("other datatypes cannot be compared"),
            })
            .collect();
        let current = values[0];
        let other = values[1];
        if current.is_none() && other.is_none() {
            return true;
        } else if current.is_none() || other.is_none() {
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
    Id,
}
