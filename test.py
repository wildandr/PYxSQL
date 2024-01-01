import mysql.connector

try:
    conn = mysql.connector.connect(
        host='localhost',
        user='root2',
        passwd='Makanmakan3x*',
        port=3306  # Sesuaikan jika port berbeda
    )
    if conn.is_connected():
        print('Koneksi berhasil')
        conn.close()
except mysql.connector.Error as e:
    print('Error:', e)