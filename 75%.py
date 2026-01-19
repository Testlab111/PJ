import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. โหลดข้อมูล
file_path = r"C:\Users\modern\OneDrive - BUU\Documents\Data Project\ชุดข้อมูลทั้งหมด 3 ปี.xlsx"

try:
    df = pd.read_excel(file_path)
except Exception:
    df = pd.read_csv(file_path.replace('.xlsx', '.csv'), encoding='cp874')

# 2. การเตรียมข้อมูล (Cleaning) - แก้ไขจุดนี้เพื่อกัน Error
# ลบคอลัมน์ว่าง
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
# ลบแถวที่มีค่าว่างในคอลัมน์สำคัญทิ้งไปเลย
df = df.dropna(subset=['Day (Saturday or Sunday)', 'Traffic condition (Red or Green)'])

df.columns = df.columns.str.strip()

# --- จุดสำคัญ: บังคับให้เป็น String ทั้งหมดเพื่อแก้ TypeError ---
df['Day (Saturday or Sunday)'] = df['Day (Saturday or Sunday)'].astype(str)
df['Traffic condition (Red or Green)'] = df['Traffic condition (Red or Green)'].astype(str)

# ฟังก์ชันแปลงเวลา
def time_to_float(time_val):
    time_str = str(time_val).strip()
    try:
        # กรณีมาเป็น HH:MM:SS
        h, m, s = map(int, time_str.split(':'))
        return h + m/60.0
    except:
        try:
            # กรณีมาเป็น HH:MM
            h, m = map(int, time_str.split(':'))
            return h + m/60.0
        except:
            return 0

df['Departure_Num'] = df['Departure'].apply(time_to_float)

# 3. การใช้ LabelEncoder (ตอนนี้จะไม่พังแล้วเพราะเป็น String ทั้งหมด)
le_day = LabelEncoder()
le_traffic = LabelEncoder()

df['Day_Encoded'] = le_day.fit_transform(df['Day (Saturday or Sunday)'])
df['Traffic_Encoded'] = le_traffic.fit_transform(df['Traffic condition (Red or Green)'])

# 4. เลือก Feature และ Target
X = df[['Day_Encoded', 'Departure_Num', 'min', 'max', 'avg']]
y = df['Traffic_Encoded']

# 5. แบ่งข้อมูลเทรน 80%
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. สร้างและเทรนโมเดล
model = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=42)
model.fit(X_train, y_train)

# 7. บันทึกโมเดล
joblib.dump(model, 'decision_tree_model.pkl')
joblib.dump(le_day, 'le_day.pkl')
joblib.dump(le_traffic, 'le_traffic.pkl')

print("--- แก้ไข Error และเทรนสำเร็จแล้ว ---")
print(f"ค่าในคอลัมน์วัน: {le_day.classes_}")
print(f"ความแม่นยำ (Validation Accuracy): {model.score(X_val, y_val):.2%}")