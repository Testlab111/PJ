import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. โหลดโมเดลและตัวแปลงที่บันทึกไว้
try:
    model = joblib.load('decision_tree_model.pkl')
    le_day = joblib.load('le_day.pkl')
    le_traffic = joblib.load('le_traffic.pkl')
except FileNotFoundError:
    print("Error: ไม่พบไฟล์โมเดล กรุณารันโค้ดส่วน Training ก่อนครับ")
    exit()

# 2. โหลดข้อมูลไฟล์ปี 68
file_path_test = r"C:\Users\modern\OneDrive - BUU\Documents\Data Project\ปี 68 .xlsx"

try:
    df_test = pd.read_excel(file_path_test)
except Exception:
    df_test = pd.read_csv(file_path_test.replace('.xlsx', '.csv'), encoding='cp874')

# 3. เตรียมข้อมูล (Cleaning & Transformation)
df_test = df_test.loc[:, ~df_test.columns.str.contains('^Unnamed')]
df_test = df_test.dropna(subset=['Day (Saturday or Sunday)', 'Traffic condition (Red or Green)'])
df_test.columns = df_test.columns.str.strip()

# บังคับเป็น String ป้องกัน TypeError
df_test['Day (Saturday or Sunday)'] = df_test['Day (Saturday or Sunday)'].astype(str)
df_test['Traffic condition (Red or Green)'] = df_test['Traffic condition (Red or Green)'].astype(str)

# ฟังก์ชันแปลงเวลา
def time_to_float(time_val):
    time_str = str(time_val).strip()
    try:
        parts = time_str.split(':')
        h = int(parts[0])
        m = int(parts[1])
        return h + m/60.0
    except:
        return 0

df_test['Departure_Num'] = df_test['Departure'].apply(time_to_float)

# 4. แปลงข้อมูลหมวดหมู่ (ใช้ transform เท่านั้น ห้ามใช้ fit_transform)
try:
    X_test = pd.DataFrame({
        'Day_Encoded': le_day.transform(df_test['Day (Saturday or Sunday)']),
        'Departure_Num': df_test['Departure_Num'],
        'min': df_test['min'],
        'max': df_test['max'],
        'avg': df_test['avg']
    })
    y_actual = le_traffic.transform(df_test['Traffic condition (Red or Green)'])
except ValueError as e:
    print(f"Error: พบข้อมูลในไฟล์ปี 68 ที่ไม่เคยปรากฏในตอนเทรน: {e}")
    exit()

# 5. ทำนายผล
y_pred = model.predict(X_test)

# 6. แสดงผลลัพธ์
print("=== สรุปผลการทำนายข้อมูลปี 68 ===")
print(f"Accuracy Score: {accuracy_score(y_actual, y_pred):.2%}")
print("\nClassification Report:")
print(classification_report(y_actual, y_pred, target_names=le_traffic.classes_))

# 7. บันทึกผลออกเป็น Excel เพื่อตรวจดูรายแถว
df_test['Predicted_Traffic'] = le_traffic.inverse_transform(y_pred)
df_test.to_excel('สรุปผลการทำนาย_ปี68.xlsx', index=False)
print("\nบันทึกไฟล์ 'สรุปผลการทำนาย_ปี68.xlsx' เรียบร้อยแล้ว")

# 8. สร้าง Confusion Matrix (Visual)
cm = confusion_matrix(y_actual, y_pred)
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=le_traffic.classes_,
            yticklabels=le_traffic.classes_)
plt.title('Confusion Matrix: Prediction vs Actual (Year 68)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# --- ส่วนที่เพิ่มต่อท้ายไฟล์เทสเดิม ---

# 1. เตรียมข้อมูลสำหรับ Heatmap (ใช้ผลลัพธ์จากการทำนาย y_pred)
df_test['Predicted_Traffic_Label'] = df_test['Predicted_Traffic'] # จากที่เราแปลงกลับเป็น Red/Green แล้ว
df_test['Is_Red'] = (df_test['Predicted_Traffic_Label'] == 'Red').astype(int)

# 2. สร้าง Pivot Table เพื่อจัดกลุ่ม วัน และ เวลา
# เราจะหาค่าเฉลี่ยความน่าจะเป็นที่จะเกิดรถติดในแต่ละช่วงเวลา
heatmap_data = df_test.pivot_table(index='Departure',
                                    columns='Day (Saturday or Sunday)',
                                    values='Is_Red',
                                    aggfunc='mean')

# 3. ตั้งค่าฟอนต์และวาด Heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.set(font='Tahoma') # หรือฟอนต์ภาษาไทยอื่นๆ ในเครื่องคุณ

sns.heatmap(heatmap_data,
            annot=True,      # แสดงตัวเลขในช่อง
            cmap='YlOrRd',   # ไล่สีจากเหลืองไปแดง (แดง = ติดมาก)
            fmt='.2f',       # ทศนิยม 2 ตำแหน่ง
            cbar_kws={'label': 'โอกาสการเกิดรถติด (0-1)'})

plt.title('Heatmap สรุปช่วงเวลาที่เสี่ยงรถติดในปี 2568 (Predict Result)')
plt.xlabel('วันหยุด')
plt.ylabel('เวลาที่ออกเดินทาง')

# บันทึกเป็นรูปภาพแยกต่างหาก
plt.savefig('traffic_heatmap_2025.png', dpi=300, bbox_inches='tight')
plt.show()

print("สร้าง Heatmap เรียบร้อยแล้ว! ดูไฟล์ภาพได้ที่ 'traffic_heatmap_2025.png'")

# --- ส่วนวิเคราะห์ข้อผิดพลาด (Error Analysis) ---

# 1. สร้าง Column เพื่อเช็คว่าทำนายถูกหรือผิด
df_test['Actual_Traffic_Label'] = df_test['Traffic condition (Red or Green)']
df_test['Is_Correct'] = (df_test['Predicted_Traffic'] == df_test['Actual_Traffic_Label'])

# 2. กรองเฉพาะรายการที่ทำนาย "ผิด"
df_errors = df_test[df_test['Is_Correct'] == False].copy()

print(f"\n=== วิเคราะห์รายการที่ทำนายผิด (จำนวน {len(df_errors)} รายการ) ===")

if not df_errors.empty:
    # แสดง 10 รายการแรกที่ทำนายผิด
    print(df_errors[['Day (Saturday or Sunday)', 'Departure', 'Actual_Traffic_Label', 'Predicted_Traffic']].head(10))

    # 3. สร้าง Heatmap เฉพาะจุดที่ทำนายผิด เพื่อดูว่ากระจุกตัวที่ช่วงเวลาไหน
    error_pivot = df_errors.pivot_table(index='Departure',
                                        columns='Day (Saturday or Sunday)',
                                        values='Is_Correct',
                                        aggfunc='count').fillna(0)

    plt.figure(figsize=(10, 6))
    sns.heatmap(error_pivot, annot=True, fmt='g', cmap='Reds')
    plt.title('Distribution of Prediction Errors (Count)')
    plt.xlabel('Day')
    plt.ylabel('Departure Time')
    plt.show()

    # 4. วิเคราะห์ว่าส่วนใหญ่ผิดที่ Case ไหน (เช่น จริงๆ รถติด แต่ทายว่าไม่ติด)
    df_errors['Error_Type'] = 'Actual: ' + df_errors['Actual_Traffic_Label'] + ' | Predicted: ' + df_errors[
        'Predicted_Traffic']
    error_summary = df_errors['Error_Type'].value_counts()

    print("\nสรุปประเภทการทำนายผิด:")
    print(error_summary)

    # พล็อตกราฟแท่งแสดงช่วงเวลาที่ผิดบ่อยที่สุด 5 อันดับแรก
    plt.figure(figsize=(8, 4))
    df_errors['Departure'].value_counts().head(5).plot(kind='bar', color='salmon')
    plt.title('Top 5 Times with Most Prediction Errors')
    plt.ylabel('Number of Errors')
    plt.xlabel('Time')
    plt.xticks(rotation=45)
    plt.show()
else:
    print("ยินดีด้วย! โมเดลทำนายถูกต้อง 100% ไม่พบข้อผิดพลาด")