from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
import pandas as pd
import os
from django.conf import settings
import matplotlib.pyplot as plt
import numpy as np
import base64
import io

def trangchu(request):
    return render(request, 'csvapp/trangchu.html')
from django.shortcuts import render

# Quản lý sản phẩm
def product_management(request):
    return render(request, 'csvapp/product_management.html')

# Quản lý vị trí sản phẩm
def product_location_management(request):
    return render(request, 'csvapp/product_location_management.html')


# Quản lý bảo hành sản phẩm
def warranty_management(request):
    return render(request, 'csvapp/warranty_management.html')
import os
import pandas as pd
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect
from .models import Report, SaleReport
from django.conf import settings
from datetime import datetime
import os
import pandas as pd
from datetime import datetime
from django.conf import settings
from django.shortcuts import render, redirect
from .models import Report, SaleReport
from django.core.files.storage import FileSystemStorage

def upload_files(request):
    if request.method == 'POST':
        if request.FILES.getlist('csv_files'):
            files = request.FILES.getlist('csv_files')
            
            # Đường dẫn thư mục 'reports'
            report_files_path = os.path.join(settings.BASE_DIR, 'reports')  # Đảm bảo thư mục reports nằm trong thư mục gốc của dự án
            os.makedirs(report_files_path, exist_ok=True)
            
            for csv_file in files:
                # Lưu file vào thư mục reports
                filename = csv_file.name
                base, extension = os.path.splitext(filename)
                counter = 1
                while os.path.exists(os.path.join(report_files_path, filename)):
                    filename = f"{base}_{counter}{extension}"
                    counter += 1
                
                # Lưu file vào thư mục reports
                file_path = os.path.join(report_files_path, filename)
                with open(file_path, 'wb') as f:
                    for chunk in csv_file.chunks():
                        f.write(chunk)
                
                # Tạo bản ghi Report trong cơ sở dữ liệu
                report = Report(file=file_path)
                report.save()

                # Đọc file CSV, chỉ lấy các cột cần thiết
                df = pd.read_csv(file_path, usecols=['Order ID', 'Product', 'Quantity Ordered', 
                                                     'Price Each', 'Order Date', 'Purchase Address'])

                # Làm sạch và chuyển đổi dữ liệu cùng lúc
                df['Quantity Ordered'] = pd.to_numeric(df['Quantity Ordered'], errors='coerce').fillna(0).astype(int)
                df['Price Each'] = pd.to_numeric(df['Price Each'], errors='coerce').fillna(0).astype(float)
                df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce').fillna(datetime(1900, 1, 1))

                # Tạo danh sách các đối tượng SaleReport
                sale_reports = [
                    SaleReport(
                        report=report,
                        order_id=row['Order ID'],
                        product=row['Product'],
                        quantity_ordered=row['Quantity Ordered'],
                        price_each=row['Price Each'],
                        order_date=row['Order Date'],
                        purchase_address=row['Purchase Address']
                    )
                    for _, row in df.iterrows()
                ]
                
                # Lưu hàng loạt vào cơ sở dữ liệu
                SaleReport.objects.bulk_create(sale_reports, batch_size=500)

            return redirect('success')
        
    return render(request, 'csvapp/upload.html')

# Success view
def success(request):
    return render(request, 'csvapp/success.html')
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from django.shortcuts import render
from django.http import HttpResponse
from .models import SaleReport
from django.conf import settings
import os
from datetime import datetime

# Action 1 view: Monthly Sales Analysis and Chart
def action1(request):
    # Lấy tất cả các SaleReport từ cơ sở dữ liệu
    sale_reports = SaleReport.objects.all()

    # Kiểm tra nếu không có dữ liệu
    if not sale_reports:
        context = {'error_message': "No sales data found."}
        return render(request, 'csvapp/your_custom_error_template.html', context)

    # Chuyển dữ liệu SaleReport thành DataFrame
    data = {
        'Order Date': [],
        'Quantity Ordered': [],
        'Price Each': [],
    }

    for report in sale_reports:
        data['Order Date'].append(report.order_date)
        data['Quantity Ordered'].append(report.quantity_ordered)
        data['Price Each'].append(report.price_each)

    # Chuyển dữ liệu thành DataFrame
    df = pd.DataFrame(data)

    # Kiểm tra các cột cần thiết
    required_columns = ['Order Date', 'Quantity Ordered', 'Price Each']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        context = {'error_message': f"Missing columns: {', '.join(missing_columns)}"}
        return render(request, 'csvapp/your_custom_error_template.html', context)

    # Tính toán doanh thu theo tháng
    df['Sales'] = df['Quantity Ordered'] * df['Price Each']  # Tạo cột Sales
    df['Month'] = pd.to_datetime(df['Order Date']).dt.month  # Lấy tháng từ Order Date

    # Nhóm theo tháng và tính tổng doanh thu (Sales)
    sales_value = df.groupby('Month')['Sales'].sum()

    # Tạo biểu đồ
    plt.figure(figsize=(10, 6))
    plt.bar(x=sales_value.index, height=sales_value)
    plt.xlabel('Months')
    plt.ylabel('Sales in USD')
    plt.title('Monthly Sales Analysis')

    # Lưu biểu đồ vào bộ nhớ dưới dạng base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Đóng biểu đồ để giải phóng bộ nhớ
    plt.close()

    context = {
        'sales_chart': image_base64,
        'sales': sales_value.to_dict(),
        'action': 'action1'
    }

    # Kiểm tra xem người dùng có muốn tải xuống không
    if request.GET.get('download') == 'true':
        buf.seek(0)  # Đặt lại vị trí bộ đệm trước khi đọc
        response = HttpResponse(buf.getvalue(), content_type='image/png')
        response['Content-Disposition'] = 'attachment; filename="monthly_sales_chart.png"'
        return response

    return render(request, 'csvapp/action1.html', context)

import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
import os
from .models import SaleReport  # Đảm bảo đã nhập đúng model

# Action 2 view: Sales by City Analysis and Chart
def action2(request):
    # Lấy tất cả các SaleReport từ cơ sở dữ liệu
    sale_reports = SaleReport.objects.all()

    # Kiểm tra nếu không có dữ liệu
    if not sale_reports:
        context = {'error_message': "No sales data found."}
        return render(request, 'csvapp/your_custom_error_template.html', context)

    # Chuyển dữ liệu SaleReport thành DataFrame
    data = {
        'Purchase Address': [],
        'Quantity Ordered': [],
        'Price Each': [],
    }

    for report in sale_reports:
        data['Purchase Address'].append(report.purchase_address)
        data['Quantity Ordered'].append(report.quantity_ordered)
        data['Price Each'].append(report.price_each)

    # Chuyển dữ liệu thành DataFrame
    df = pd.DataFrame(data)

    # Kiểm tra các cột cần thiết
    required_columns = ['Purchase Address', 'Quantity Ordered', 'Price Each']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        context = {'error_message': f"Missing columns: {', '.join(missing_columns)}"}
        return render(request, 'csvapp/your_custom_error_template.html', context)

    # Tính toán doanh thu theo thành phố
    df['Sales'] = df['Quantity Ordered'] * df['Price Each']  # Tạo cột Sales

    # Trích xuất thành phố từ 'Purchase Address'
    address_to_city = lambda address: address.split(',')[1].strip() if pd.notna(address) and len(address.split(',')) > 1 else 'Unknown'
    df['City'] = df['Purchase Address'].apply(address_to_city)

    # Nhóm theo thành phố và tính tổng doanh thu (Sales)
    sales_value_city = df.groupby('City')['Sales'].sum()

    # Tạo biểu đồ doanh thu theo thành phố
    plt.figure(figsize=(10, 6))
    plt.bar(x=sales_value_city.index.tolist(), height=sales_value_city)
    plt.xticks(rotation=90, size=8)
    plt.xlabel('Cities')
    plt.ylabel('Sales in USD')
    plt.title('Sales by City')

    # Lưu biểu đồ vào bộ nhớ dưới dạng base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Đóng biểu đồ để giải phóng bộ nhớ
    plt.close()

    context = {
        'sales_chart': image_base64,
        'sales': sales_value_city.to_dict(),
        'action': 'action2'
    }

    # Kiểm tra xem người dùng có muốn tải xuống không
    if request.GET.get('download') == 'true':
        buf.seek(0)  # Đặt lại vị trí bộ đệm trước khi đọc
        response = HttpResponse(buf.getvalue(), content_type='image/png')
        response['Content-Disposition'] = 'attachment; filename="sales_by_city_chart.png"'
        return response

    return render(request, 'csvapp/action2.html', context)
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from django.shortcuts import render
from django.http import HttpResponse
from .models import SaleReport  # Đảm bảo đã nhập đúng model

def action3(request):
    # Lấy tất cả các SaleReport từ cơ sở dữ liệu
    sale_reports = SaleReport.objects.all()

    # Kiểm tra nếu không có dữ liệu
    if not sale_reports:
        context = {'error_message': "No sales data found."}
        return render(request, 'csvapp/your_custom_error_template.html', context)

    # Chuyển dữ liệu SaleReport thành DataFrame
    data = {
        'Order Date': [],
        'Quantity Ordered': [],
        'Price Each': [],
    }

    for report in sale_reports:
        data['Order Date'].append(report.order_date)
        data['Quantity Ordered'].append(report.quantity_ordered)
        data['Price Each'].append(report.price_each)

    # Chuyển dữ liệu thành DataFrame
    df = pd.DataFrame(data)

    # Kiểm tra các cột cần thiết
    required_columns = ['Order Date', 'Quantity Ordered', 'Price Each']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        context = {'error_message': f"Missing columns: {', '.join(missing_columns)}"}
        return render(request, 'csvapp/your_custom_error_template.html', context)

    # Chuyển 'Order Date' thành datetime và trích xuất giờ
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df['Hours'] = df['Order Date'].dt.hour

    # Tính toán 'Sales'
    df['Quantity Ordered'] = pd.to_numeric(df['Quantity Ordered'], errors='coerce')
    df['Price Each'] = pd.to_numeric(df['Price Each'], errors='coerce')
    df['Sales'] = df['Quantity Ordered'] * df['Price Each']

    # Xóa các cột không cần thiết
    df = df.drop(columns=['Order Date'], errors='ignore')

    # Nhóm theo giờ và tính tổng 'Sales'
    sales_value_hours = df.groupby('Hours').sum()['Sales']

    # Kiểm tra xem người dùng có muốn tải biểu đồ về dưới dạng PNG không
    if 'download_image' in request.GET:
        plt.figure(figsize=(10, 6))
        plt.plot(sales_value_hours.index.tolist(), sales_value_hours, marker='o')
        plt.grid()
        plt.xticks(rotation=90, size=8)
        plt.xlabel('Hours')
        plt.ylabel('Sales in USD')
        plt.title('Sales by Hour')

        response = HttpResponse(content_type='image/png')
        response['Content-Disposition'] = 'attachment; filename="sales_by_hour_chart.png"'
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        response.write(buf.getvalue())
        buf.close()
        return response

    # Tạo biểu đồ doanh thu theo giờ
    plt.figure(figsize=(10, 6))
    plt.plot(sales_value_hours.index.tolist(), sales_value_hours, marker='o')
    plt.grid()
    plt.xticks(rotation=90, size=8)
    plt.xlabel('Hours')
    plt.ylabel('Sales in USD')
    plt.title('Sales by Hour')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    context = {
        'sales_chart': image_base64,
        'sales_by_hour': sales_value_hours.to_dict()
    }

    return render(request, 'csvapp/action3.html', context)
def action4(request):
    # Lấy tất cả các SaleReport từ cơ sở dữ liệu
    sale_reports = SaleReport.objects.all()

    # Kiểm tra nếu không có dữ liệu
    if not sale_reports:
        context = {'error_message': "No sales data found."}
        return render(request, 'csvapp/your_custom_error_template.html', context)

    # Chuyển dữ liệu SaleReport thành DataFrame
    data = {
        'Product': [],
        'Quantity Ordered': [],
        'Price Each': [],
    }

    for report in sale_reports:
        # Ensure 'product' is a Product instance, not a string
        data['Product'].append(report.product)
        data['Quantity Ordered'].append(report.quantity_ordered)
        data['Price Each'].append(report.price_each)

    # Chuyển dữ liệu thành DataFrame
    df = pd.DataFrame(data)

    # Kiểm tra các cột cần thiết
    required_columns = ['Product', 'Quantity Ordered', 'Price Each']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        context = {'error_message': f"Missing columns: {', '.join(missing_columns)}"}
        return render(request, 'csvapp/your_custom_error_template.html', context)

    # Kiểm tra và chuyển đổi các cột sang kiểu số nếu cần thiết
    df['Quantity Ordered'] = pd.to_numeric(df['Quantity Ordered'], errors='coerce')
    df['Price Each'] = pd.to_numeric(df['Price Each'], errors='coerce')

    # Tính toán doanh thu theo sản phẩm
    df['Sales'] = df['Quantity Ordered'] * df['Price Each']  # Tạo cột Sales

    # Nhóm theo sản phẩm và tính tổng doanh thu (Sales)
    top_products = df.groupby('Product').sum()['Sales'].nlargest(10)

    # Lấy danh sách sản phẩm và doanh thu
    products = top_products.index.tolist()
    sales_values = top_products.tolist()

    # Tạo biểu đồ doanh thu theo sản phẩm
    plt.figure(figsize=(10, 6))
    plt.bar(products, sales_values)
    plt.xticks(rotation=90, size=8)
    plt.xlabel('Products')
    plt.ylabel('Sales in USD')
    plt.title('Top 10 Products by Sales')

    # Lưu biểu đồ vào bộ nhớ dưới dạng base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Đóng biểu đồ để giải phóng bộ nhớ
    plt.close()

    context = {
        'product_combination_chart': image_base64,
        'top_combinations': top_products.to_dict(),
        'action': 'action4'
    }

    # Kiểm tra xem người dùng có muốn tải xuống không
    if request.GET.get('download') == 'true':
        buf.seek(0)  # Đặt lại vị trí bộ đệm trước khi đọc
        response = HttpResponse(buf.getvalue(), content_type='image/png')
        response['Content-Disposition'] = 'attachment; filename="top_10_products_chart.png"'
        return response

    return render(request, 'csvapp/action4.html', context)


import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from django.http import HttpResponse
from django.shortcuts import render
from .models import SaleReport
from django.conf import settings

def action5(request):
    # Lấy tất cả dữ liệu từ SaleReport
    sale_reports = SaleReport.objects.all()
    
    if not sale_reports:
        context = {
            'error_message': "No sales data found."
        }
        return render(request, 'csvapp/your_custom_error_template.html', context)
    
    # Chuyển dữ liệu SaleReport thành DataFrame
    data = {
        'Product': [],
        'Quantity Ordered': [],
        'Price Each': [],
    }
    
    for report in sale_reports:
        # Ensure 'product' is a Product instance, not a string
        data['Product'].append(report.product)
        data['Quantity Ordered'].append(report.quantity_ordered)
        data['Price Each'].append(report.price_each)
    
    # Chuyển dữ liệu thành DataFrame
    df = pd.DataFrame(data)
    
    # Kiểm tra sự tồn tại của các cột cần thiết
    required_columns = ['Product', 'Quantity Ordered', 'Price Each']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        context = {
            'error_message': f"Missing columns: {', '.join(missing_columns)}"
        }
        return render(request, 'csvapp/your_custom_error_template.html', context)
    
    # Chuyển đổi kiểu dữ liệu
    df['Quantity Ordered'] = pd.to_numeric(df['Quantity Ordered'], errors='coerce')
    df['Price Each'] = pd.to_numeric(df['Price Each'], errors='coerce')
    
    # Xử lý dữ liệu
    df = df.dropna(subset=['Quantity Ordered', 'Price Each'])
    
    grouped = df.groupby('Product').agg({
        'Quantity Ordered': 'sum',
        'Price Each': 'mean'
    }).reset_index()
    
    all_products = grouped.set_index('Product')['Quantity Ordered']
    prices = grouped.set_index('Product')['Price Each']
    products_ls = grouped['Product'].tolist()

    x = products_ls
    y1 = all_products
    y2 = prices

    # Check if the user wants to download the chart as PNG
    if 'download_image' in request.GET:
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.bar(x, y1, color='g', alpha=0.6, label='Quantity Ordered')
        ax2.plot(x, y2, 'b-', marker='o', label='Price Each')

        ax1.set_xticklabels(products_ls, rotation=90, size=8)
        ax1.set_xlabel('Products')
        ax1.set_ylabel('Quantity Ordered', color='g')
        ax2.set_ylabel('Price Each', color='b')

        response = HttpResponse(content_type='image/png')
        response['Content-Disposition'] = 'attachment; filename="product_quantity_and_price_chart.png"'
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        response.write(buffer.getvalue())
        buffer.close()
        return response

    # Otherwise, render the chart on the webpage
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.bar(x, y1, color='g', alpha=0.6, label='Quantity Ordered')
    ax2.plot(x, y2, 'b-', marker='o', label='Price Each')

    ax1.set_xticklabels(products_ls, rotation=90, size=8)
    ax1.set_xlabel('Products')
    ax1.set_ylabel('Quantity Ordered', color='g')
    ax2.set_ylabel('Price Each', color='b')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    context = {
        'product_price_chart': image_base64
    }
    
    return render(request, 'csvapp/action5.html', context)

from django.shortcuts import render
from .models import SaleReport  # Import SaleReport model

def search_orders_view(request):
    # Lấy từ khóa tìm kiếm từ request GET
    query = request.GET.get('query', '')

    # Khởi tạo biến kết quả là None
    search_result = None

    if query:
        try:
            # Chuyển từ khóa thành số nguyên và tìm đơn hàng theo ID
            report_id = int(query)
            search_result = SaleReport.objects.filter(id=report_id).first()  # Lấy đơn hàng đầu tiên phù hợp
        except ValueError:
            search_result = None  # Nếu từ khóa không hợp lệ, không trả về kết quả

    context = {
        'search_result': search_result,  # Gửi kết quả đơn hàng hoặc None
        'query': query,  # Gửi từ khóa tìm kiếm để hiển thị lại trên form
    }

    return render(request, 'csvapp/search_orders_view.html', context)


from django.shortcuts import render
from .models import SaleReport  # Import SaleReport model
import pandas as pd

def search_orders_date(request):
    # Lấy tất cả các báo cáo bán hàng từ cơ sở dữ liệu
    sale_reports = SaleReport.objects.all()

    # Chuyển dữ liệu từ queryset thành DataFrame
    df = pd.DataFrame(list(sale_reports.values('order_date', 'product', 'quantity_ordered', 'price_each', 'purchase_address')))

    # Loại bỏ khoảng trắng thừa trong tên cột
    df.columns = df.columns.str.strip()

    # Kiểm tra cột 'Order Date'
    if 'order_date' not in df.columns:
        return render(request, 'csvapp/search_orders_date.html', {'orders': []})

    # Chuyển đổi cột 'Order Date' sang định dạng datetime
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

    # Lấy giá trị từ form
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')

    if start_date and end_date:
        try:
            # Chuyển đổi ngày bắt đầu và ngày kết thúc thành định dạng datetime
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)

            # Thực hiện lọc dữ liệu dựa trên khoảng thời gian
            filtered_df = df[(df['order_date'] >= start_date) & (df['order_date'] <= end_date)]
        except Exception as e:
            filtered_df = pd.DataFrame()  # Gán một DataFrame rỗng nếu có lỗi
    else:
        filtered_df = df

    # Chuyển đổi DataFrame thành danh sách từ điển để gửi tới template
    orders = filtered_df.rename(columns={
        'order_date': 'Order Date',
        'product': 'Product',
        'quantity_ordered': 'Quantity Ordered',
        'price_each': 'Price Each',
        'purchase_address': 'Purchase Address'
    }).to_dict('records') if not filtered_df.empty else []

    return render(request, 'csvapp/search_orders_date.html', {'orders': orders})
import pandas as pd
import os
from django.conf import settings
from django.shortcuts import render

def suggest_products(request):
    product_name = request.GET.get('product_name')
    suggestions = []
    total_products = 0  # Biến để đếm tổng số sản phẩm

    # Đường dẫn tới file CSV
    csv_file_path = os.path.join(settings.MEDIA_ROOT, 'combined_sales_data.csv')

    try:
        # Đọc file CSV vào DataFrame
        df = pd.read_csv(csv_file_path)

        # In ra danh sách các cột để kiểm tra
        print("Danh sách các cột trong DataFrame:", df.columns)

        # Thay đổi tên cột để loại bỏ dấu cách
        df.columns = [col.replace(' ', '_') for col in df.columns]

        # Kiểm tra sự tồn tại của cột 'Price_Each' và 'Quantity_Ordered'
        if 'Price_Each' not in df.columns:
            print("Cột 'Price_Each' không tồn tại trong dữ liệu.")
            df['Price_Each'] = 0  # Thêm cột 'Price_Each' nếu không có, và gán giá trị mặc định bằng 0

        if 'Quantity_Ordered' not in df.columns:
            print("Cột 'Quantity_Ordered' không tồn tại trong dữ liệu.")
            df['Quantity_Ordered'] = 0  # Thêm cột 'Quantity_Ordered' nếu không có, và gán giá trị mặc định bằng 0

        # Đảm bảo rằng các cột 'Price_Each' và 'Quantity_Ordered' là kiểu số
        df['Price_Each'] = pd.to_numeric(df['Price_Each'], errors='coerce')
        df['Quantity_Ordered'] = pd.to_numeric(df['Quantity_Ordered'], errors='coerce')

        # Đếm tổng số sản phẩm khác nhau trong DataFrame
        total_products = df['Product'].nunique()  # Số sản phẩm khác nhau

        if product_name:
            # Gộp các sản phẩm lại và đếm số lượng
            grouped_suggestions = df.groupby('Product').agg(
                Total_Quantity=('Quantity_Ordered', 'sum'),
                Total_Sales=('Price_Each', 'sum')
            ).reset_index()

            # Tìm kiếm các sản phẩm liên quan
            suggestions = grouped_suggestions[grouped_suggestions['Product'].str.contains(product_name, case=False, na=False)]

            # Chuyển đổi DataFrame thành danh sách từ điển
            if suggestions is not None and not suggestions.empty:
                suggestions = suggestions.to_dict(orient='records')

    except Exception as e:
        print(f"Error reading CSV file: {e}")  # In ra lỗi nếu có

    context = {'suggestions': suggestions, 'total_products': total_products}
    return render(request, 'csvapp/suggest_products.html', context)

from django.shortcuts import render, get_object_or_404, redirect
from .models import Product
from .form import ProductForm
from django.contrib.auth.decorators import login_required, user_passes_test
from django.shortcuts import render, get_object_or_404, redirect
from django.db.models import Q
from django.contrib import messages
from .models import Product
from .form import ProductForm
def product_list(request):
    # Lấy từ khóa tìm kiếm từ GET request
    search_query = request.GET.get('search', '')
    
    # Tìm kiếm sản phẩm nếu có từ khóa
    if search_query:
        products = Product.objects.filter(
            Q(name__icontains=search_query) | Q(description__icontains=search_query)
        )
    else:
        products = Product.objects.all()  # Nếu không có từ khóa tìm kiếm thì lấy tất cả sản phẩm

    return render(request, 'product_list.html', {'products': products, 'search_query': search_query})
# Kiểm tra quyền admin
def is_admin(user):
    return user.is_superuser

# Hiển thị danh sách sản phẩm (tất cả người dùng đăng nhập đều được phép)
@login_required
def product_management(request):
    search_query = request.GET.get('search', '')
    if search_query:
        products = Product.objects.filter(Q(name__icontains=search_query) | Q(description__icontains=search_query))
    else:
        products = Product.objects.all()
    return render(request, 'csvapp/product_management.html', {'products': products})

# Xem chi tiết sản phẩm (tất cả người dùng đăng nhập đều được phép)
@login_required
def product_detail(request, id):
    product = get_object_or_404(Product, id=id)
    return render(request, 'csvapp/product_detail.html', {'product': product})

# Thêm sản phẩm (chỉ dành cho admin)
@login_required
@user_passes_test(is_admin, login_url='no_permission')
def add_product(request):
    if request.method == 'POST':
        form = ProductForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Sản phẩm mới đã được thêm thành công.')
            return redirect('product_management')
    else:
        form = ProductForm()
    return render(request, 'csvapp/add_product.html', {'form': form})

# Sửa sản phẩm (chỉ dành cho admin)
@login_required
@user_passes_test(is_admin, login_url='no_permission')
def edit_product(request, product_id):
    product = get_object_or_404(Product, id=product_id)
    if request.method == 'POST':
        form = ProductForm(request.POST, instance=product)
        if form.is_valid():
            form.save()
            messages.success(request, f'Sản phẩm "{product.name}" đã được cập nhật.')
            return redirect('product_management')
    else:
        form = ProductForm(instance=product)
    return render(request, 'csvapp/edit_product.html', {'form': form})

# Xóa sản phẩm (chỉ dành cho admin)
@login_required
@user_passes_test(is_admin, login_url='no_permission')
def delete_product(request, product_id):
    product = get_object_or_404(Product, id=product_id)
    if request.method == 'POST':
        product.delete()
        messages.success(request, f'Sản phẩm "{product.name}" đã được xóa thành công.')
        return redirect('product_management')
    return render(request, 'csvapp/delete_product.html', {'product': product})

from django.shortcuts import render, get_object_or_404, redirect
from .models import Location
from .form import LocationForm

from django.shortcuts import render, get_object_or_404, redirect
from .models import Location
from .form import LocationForm

# Hiển thị danh sách vị trí
def location_management(request):
    locations = Location.objects.all()  # Lấy tất cả các vị trí
    return render(request, 'csvapp/location_management.html', {'locations': locations})

# Thêm vị trí
def add_location(request):
    if request.method == 'POST':
        form = LocationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('location_management')
    else:
        form = LocationForm()
    return render(request, 'csvapp/add_location.html', {'form': form})

from django.shortcuts import render, get_object_or_404, redirect
from .models import Location
from .form import LocationForm

# Edit an existing location
def edit_location(request, location_id):
    location = get_object_or_404(Location, id=location_id)
    if request.method == 'POST':
        form = LocationForm(request.POST, instance=location)
        if form.is_valid():
            form.save()
            return redirect('location_management')
    else:
        form = LocationForm(instance=location)
    return render(request, 'csvapp/edit_location.html', {'form': form, 'location': location})

from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from .models import Location

# Delete a location
from django.shortcuts import render, get_object_or_404, redirect
from .models import Location

def delete_location(request, location_id):
    location = get_object_or_404(Location, id=location_id)
    if request.method == 'POST':
        location.delete()
        return redirect('location_management')  # Quay lại trang quản lý vị trí
    return render(request, 'csvapp/delete_location.html', {'location': location})


from django.shortcuts import render, get_object_or_404, redirect
from .models import Product
from .form import ProductForm

# Edit an existing product
def edit_product(request, product_id):
    product = get_object_or_404(Product, id=product_id)
    if request.method == 'POST':
        form = ProductForm(request.POST, instance=product)
        if form.is_valid():
            form.save()
            return redirect('product_management')
    else:
        form = ProductForm(instance=product)
    return render(request, 'csvapp/edit_product.html', {'form': form, 'product': product})

from django.shortcuts import render, redirect, get_object_or_404
from .models import Supplier
from .form import SupplierForm

def supplier_management(request):
    suppliers = Supplier.objects.all()  # Lấy danh sách tất cả nhà cung cấp
    return render(request, 'csvapp/supplier_management.html', {'suppliers': suppliers})

# views.py
from django.shortcuts import render, redirect
from .form import SupplierForm

from django.shortcuts import render, redirect
from .form import SupplierForm  # Giả sử bạn có một lớp form để xử lý

def add_supplier(request):
    if request.method == 'POST':
        form = SupplierForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('supplier_management')  # Chuyển hướng đến trang quản lý nhà cung cấp
    else:
        form = SupplierForm()  # Hiển thị form trống khi không phải là POST

    return render(request, 'csvapp/add_supplier.html', {'form': form})  # Render template với form


def edit_supplier(request, supplier_id):
    supplier = get_object_or_404(Supplier, id=supplier_id)
    if request.method == 'POST':
        form = SupplierForm(request.POST, instance=supplier)
        if form.is_valid():
            form.save()
            return redirect('supplier_management')  # Chuyển hướng đến trang quản lý nhà cung cấp
    else:
        form = SupplierForm(instance=supplier)
    return render(request, 'csvapp/edit_supplier.html', {'form': form})

def delete_supplier(request, supplier_id):
    supplier = get_object_or_404(Supplier, id=supplier_id)
    if request.user.is_superuser:  # Check if the user is a superuser
        if request.method == 'POST':
            supplier.delete()
            return redirect('supplier_management')  # Redirect to the supplier management page
    else:
        return redirect('supplier_management')  # In case the user is not a superuser

# views.py
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from .models import Product, ProductImage
from .form import ProductImageForm

from django.shortcuts import render, get_object_or_404
from .models import Product, ProductImage
from .form import ProductImageForm

def add_product_image(request, product_id):
    product = get_object_or_404(Product, id=product_id)
    images = ProductImage.objects.filter(product=product)  # Lấy danh sách hình ảnh của sản phẩm

    if request.method == 'POST':
        form = ProductImageForm(request.POST, request.FILES)
        if form.is_valid():
            product_image = form.save(commit=False)
            product_image.product = product  # Gán sản phẩm cho hình ảnh
            product_image.save()
            messages.success(request, "Hình ảnh đã được thêm thành công!")
            return redirect('product_management')  # Quay lại trang hiện tại
    else:
        form = ProductImageForm()

    return render(request, 'csvapp/add_product_image.html', {
        'form': form,
        'product': product,
        'images': images
    })

from django.shortcuts import get_object_or_404, render
from django.http import JsonResponse
from .models import Product, ProductImage

from django.shortcuts import get_object_or_404, redirect
from .models import ProductImage

def delete_product_image(request, product_id, image_id):
    # Get the image object to delete
    image = get_object_or_404(ProductImage, id=image_id, product_id=product_id)
    image.delete()  # Delete the image

    # Redirect back to the product management page or wherever needed
    return redirect('product_management')  # Update this to the appropriate URL name


from django.shortcuts import render, redirect, get_object_or_404
from .models import Warranty, Product
from .form import WarrantyForm  # Tạo form trong bước 3

def warranty_list(request):
    warranties = Warranty.objects.all()
    return render(request, 'csvapp/warranty_list.html', {'warranties': warranties})

def add_warranty(request):
    if request.method == 'POST':
        form = WarrantyForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('warranty_list')  # Redirect về danh sách bảo hành
    else:
        form = WarrantyForm()
    return render(request, 'csvapp/add_warranty.html', {'form': form})

def edit_warranty(request, warranty_id):
    warranty = get_object_or_404(Warranty, id=warranty_id)
    if request.method == 'POST':
        form = WarrantyForm(request.POST, instance=warranty)
        if form.is_valid():
            form.save()
            return redirect('warranty_list')  # Redirect về danh sách bảo hành
    else:
        form = WarrantyForm(instance=warranty)
    return render(request, 'csvapp/edit_warranty.html', {'form': form, 'warranty': warranty})

def delete_warranty(request, warranty_id):
    warranty = get_object_or_404(Warranty, id=warranty_id)
    warranty.delete()
    return redirect('warranty_list')  # Redirect về danh sách bảo hành
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponseForbidden
from django.contrib import messages
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from datetime import date
from .models import Product, EntryProduct, WarehouseEntry, Supplier
from .form import WarehouseEntryForm
import logging

logger = logging.getLogger(__name__)

from django.shortcuts import render, redirect
from datetime import datetime  # Để xử lý ngày giờ

def warehouse_entry(request):
    if request.method == 'POST':
        form = WarehouseEntryForm(request.POST)
        if form.is_valid():
            warehouse_entry = form.save(commit=False)
            
            # Sử dụng ngày từ form
            entry_date = request.POST.get('entry_date')
            warehouse_entry.entry_date = datetime.strptime(entry_date, '%Y-%m-%d').date()

            warehouse_entry.save()

            total_quantity = 0  # Initialize total quantity
            for key, value in request.POST.items():
                if key.startswith('product_') and value:
                    product_id = key.split('_')[1]
                    quantity = int(request.POST.get(f'quantity_{product_id}', 0))

                    if quantity > 0:
                        product = Product.objects.get(id=product_id)
                        
                        # Create EntryProduct
                        EntryProduct.objects.create(
                            warehouse_entry=warehouse_entry,
                            product=product,
                            quantity=quantity
                        )

                        # Chỉ cập nhật số lượng khi trạng thái là 'Đã nhập'
                        if warehouse_entry.status == WarehouseEntry.COMPLETED:
                            product.quantity += quantity  # Add the quantity to the current stock
                            product.save()  # Save the updated product quantity

                        total_quantity += quantity  # Add to total quantity

            # Update total_quantity for warehouse_entry
            warehouse_entry.total_quantity = total_quantity
            warehouse_entry.save()  # Save the updated total_quantity

            return redirect('warehouse_entry_list')
    else:
        form = WarehouseEntryForm()
        suppliers = Supplier.objects.all()
        return render(request, 'csvapp/warehouse_entry.html', {'form': form, 'suppliers': suppliers})

def get_supplier_products(request, supplier_id):
    supplier = get_object_or_404(Supplier, id=supplier_id)
    products = Product.objects.filter(supplier=supplier).values('id', 'name')
    return JsonResponse({'products': list(products)})


from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from .models import WarehouseEntry, EntryProduct, Supplier
from .form import WarehouseEntryForm
from django.contrib.auth.decorators import login_required


def warehouse_entry_list(request):
    entries = WarehouseEntry.objects.all()
    return render(request, 'csvapp/warehouse_entry_list.html', {'entries': entries})
from django.shortcuts import render, redirect
from .models import WarehouseEntry, EntryProduct, Product, Supplier

from django.shortcuts import render, redirect
from .models import WarehouseEntry, EntryProduct, Product, Supplier

def edit_warehouse_entry(request, entry_id):
    try:
        # Lấy phiếu nhập kho theo entry_id
        entry = WarehouseEntry.objects.get(id=entry_id)
        entry_products = EntryProduct.objects.filter(warehouse_entry=entry)
    except WarehouseEntry.DoesNotExist:
        return redirect('error_page')

    suppliers = Supplier.objects.all()
    selected_supplier = entry.supplier
    products_of_supplier = Product.objects.filter(supplier=selected_supplier)

    if request.method == 'POST':
        entry.entry_date = request.POST.get('entry_date')
        entry.supplier = Supplier.objects.get(id=request.POST.get('supplier'))
        entry.status = request.POST.get('status')

        # Lưu số lượng sản phẩm
        for product in products_of_supplier:
            quantity_key = f'quantity_{product.id}'
            quantity = request.POST.get(quantity_key, None)

            # Nếu nhập số lượng
            if quantity is not None:
                try:
                    # Lấy entry product nếu tồn tại, nếu không sẽ tạo mới
                    entry_product, created = EntryProduct.objects.get_or_create(
                        warehouse_entry=entry,
                        product=product,
                        defaults={'quantity': quantity}
                    )
                    # Cập nhật số lượng nếu entry product đã tồn tại
                    if not created:
                        entry_product.quantity = int(quantity)
                        entry_product.save()
                except ValueError:
                    pass  # Bỏ qua giá trị không hợp lệ

        # Cập nhật tổng số lượng
        entry.update_total_quantity()

        if entry.status == WarehouseEntry.COMPLETED:
            entry.update_inventory()

        return redirect('warehouse_entry_list')

    return render(request, 'csvapp/warehouse_entry_edit.html', {
        'entry': entry,
        'entry_products': entry_products,
        'suppliers': suppliers,
        'products_of_supplier': products_of_supplier,
    })


def delete_warehouse_entry(request, entry_id):
    entry = get_object_or_404(WarehouseEntry, pk=entry_id)
    
    if entry.status != 'Chưa nhập':
        messages.error(request, "Không thể xóa khi phiếu nhập không ở trạng thái 'Chưa nhập'.")
        return redirect('warehouse_entry_list')

    entry.delete()
    messages.success(request, "Phiếu nhập kho đã được xóa.")
    return redirect('warehouse_entry_list')


def update_status(request, entry_id):
    warehouse_entry = WarehouseEntry.objects.get(id=entry_id)

    if request.method == 'POST':
        new_status = request.POST.get('status')

        if new_status == 'Đã nhập' and warehouse_entry.status != 'Đã nhập':
            # Cập nhật số lượng vào kho khi trạng thái thay đổi thành "Đã nhập"
            for entry_product in warehouse_entry.entry_products.all():
                product = entry_product.product
                product.quantity += entry_product.quantity  # Cập nhật số lượng sản phẩm vào kho
                product.save()

        # Cập nhật trạng thái phiếu nhập kho
        warehouse_entry.status = new_status
        warehouse_entry.save()

        return redirect('warehouse_entry_list')
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .models import WarehouseExit, ExitProduct, Product


def update_exit_status(request, exit_id):
    # Lấy đối tượng WarehouseExit dựa trên exit_id
    warehouse_exit = get_object_or_404(WarehouseExit, id=exit_id)

    if request.method == 'POST':
        # Lấy trạng thái mới từ form
        new_status = request.POST.get('status')

        # Kiểm tra nếu trạng thái thay đổi thành "Đã xuất"
        if new_status == 'Đã xuất' and warehouse_exit.status != 'Đã xuất':
            # Cập nhật số lượng vào kho khi trạng thái thay đổi thành "Đã xuất"
            for exit_product in warehouse_exit.exit_products.all():
                product = exit_product.product
                # Giảm số lượng sản phẩm khỏi kho
                product.quantity -= exit_product.quantity
                product.save()

            # Thêm thông báo thành công
            messages.success(request, f"Đã cập nhật trạng thái thành 'Đã xuất' và giảm số lượng kho.")

        # Cập nhật trạng thái phiếu xuất kho
        warehouse_exit.status = new_status
        warehouse_exit.save()

        # Thêm thông báo trạng thái được cập nhật
        messages.success(request, f"Trạng thái phiếu xuất kho đã được cập nhật thành {new_status}.")

        return redirect('warehouse_exit_list')

    # Nếu là GET request, chuyển hướng về danh sách xuất kho
    return redirect('warehouse_exit_list')

    
def warehouse_exit_list(request):
    exits = WarehouseExit.objects.all()
    return render(request, 'csvapp/warehouse_exit_list.html', {'exits': exits})

# Hàm lấy sản phẩm từ nhà cung cấp
from django.http import JsonResponse
from .models import Product

def get_supplier_products(request, supplier_id):
    products = Product.objects.filter(supplier_id=supplier_id)
    product_list = [{'id': product.id, 'name': product.name} for product in products]
    return JsonResponse({'products': product_list})

from django.shortcuts import render, redirect
from django.contrib import messages
from django.utils import timezone
from .models import Product, WarehouseExit, ExitProduct, Supplier
from django.http import JsonResponse

from django.shortcuts import render, redirect
from django.contrib import messages
from .models import WarehouseExit, ExitProduct, Product, Supplier

def warehouse_exit(request):
    suppliers = Supplier.objects.all()  # Lấy tất cả nhà cung cấp

    if request.method == 'POST':
        exit_date = request.POST.get('exit_date')
        supplier_id = request.POST.get('supplier')  # Lấy nhà cung cấp từ form
        products = request.POST.keys()

        if not supplier_id:
            messages.error(request, "Vui lòng chọn nhà cung cấp.")
            return redirect('warehouse_exit')

        # Tạo bản ghi WarehouseExit mới
        exit_entry = WarehouseExit(exit_date=exit_date, supplier_id=supplier_id)
        exit_entry.save()

        total_quantity = 0
        for product_key in products:
            if product_key.startswith('product_'):
                product_id = product_key.split('_')[1]
                quantity = request.POST.get(f'quantity_{product_id}')
                if int(quantity) > 0:
                    product = Product.objects.get(id=product_id)

                    # Tạo bản ghi ExitProduct
                    ExitProduct.objects.create(
                        warehouse_exit=exit_entry,
                        product=product,
                        quantity=quantity
                    )

                    # Kiểm tra trạng thái trước khi giảm số lượng tồn kho
                    if exit_entry.status == 'Đã xuất':
                        product.quantity -= int(quantity)
                        product.save()

                    total_quantity += int(quantity)  # Cộng dồn số lượng

        # Cập nhật tổng số lượng xuất kho cho WarehouseExit
        exit_entry.total_quantity = total_quantity
        exit_entry.save()  # Lưu bản ghi WarehouseExit với tổng số lượng

        messages.success(request, f"Xuất kho thành công với {total_quantity} sản phẩm.")
        return redirect('warehouse_exit_list')

    return render(request, 'csvapp/warehouse_exit.html', {'suppliers': suppliers})

# View hiển thị danh sách phiếu xuất kho

# View lấy sản phẩm của nhà cung cấp
def get_supplier_products(request, supplier_id):
    try:
        supplier = Supplier.objects.get(id=supplier_id)
        products = Product.objects.filter(supplier=supplier)
        product_list = [{'id': product.id, 'name': product.name} for product in products]
        return JsonResponse({'products': product_list})
    except Supplier.DoesNotExist:
        return JsonResponse({'error': 'Nhà cung cấp không tồn tại'}, status=404)

# View cập nhật trạng thái xuất kho


def warehouse_exit_edit(request, exit_id):
    # Lấy thông tin phiếu xuất kho
    exit_entry = get_object_or_404(WarehouseExit, id=exit_id)
    exit_products = ExitProduct.objects.filter(warehouse_exit=exit_entry)

    # Lấy danh sách các nhà cung cấp và sản phẩm của nhà cung cấp đã chọn
    suppliers = Supplier.objects.all()
    selected_supplier = exit_entry.supplier
    products_of_supplier = Product.objects.filter(supplier=selected_supplier)

    if request.method == 'POST':
        # Cập nhật thông tin phiếu xuất kho
        exit_entry.exit_date = request.POST.get('exit_date')
        exit_entry.supplier = Supplier.objects.get(id=request.POST.get('supplier'))
        exit_entry.status = request.POST.get('status')

        # Cập nhật hoặc tạo mới các sản phẩm trong phiếu xuất kho
        for product in products_of_supplier:
            quantity_key = f'quantity_{product.id}'
            quantity = request.POST.get(quantity_key, None)

            if quantity is not None:
                try:
                    # Kiểm tra số lượng nhập vào có hợp lệ không
                    quantity = int(quantity)
                    if quantity < 0:
                        raise ValueError("Số lượng không thể là số âm")

                    # Cập nhật hoặc tạo mới ExitProduct
                    exit_product, created = ExitProduct.objects.get_or_create(
                        warehouse_exit=exit_entry,
                        product=product,
                        defaults={'quantity': quantity}
                    )
                    if not created:
                        exit_product.quantity = quantity
                        exit_product.save()

                except ValueError as e:
                    messages.error(request, f"Số lượng không hợp lệ cho sản phẩm {product.name}: {e}")
                    continue

        # Cập nhật tổng số lượng của phiếu xuất kho
        exit_entry.total_quantity = sum(ep.quantity for ep in exit_entry.exit_products.all())
        exit_entry.save()

        # Kiểm tra trạng thái, nếu là "Đã xuất" mới giảm số lượng tồn kho
        if exit_entry.status == WarehouseExit.COMPLETED:
            for exit_product in exit_entry.exit_products.all():
                product = exit_product.product
                if product.stock >= exit_product.quantity:
                    product.stock -= exit_product.quantity
                    product.save()
                else:
                    messages.error(request, f"Không đủ hàng tồn kho cho sản phẩm {product.name}")
                    return redirect('warehouse_exit_list')

        return redirect('warehouse_exit_list')

    return render(request, 'csvapp/warehouse_exit_edit.html', {
        'exit_entry': exit_entry,
        'exit_products': exit_products,
        'suppliers': suppliers,
        'products_of_supplier': products_of_supplier,
    })

# View xóa phiếu xuất kho

def warehouse_exit_delete(request, exit_id):
    warehouse_exit = get_object_or_404(WarehouseExit, id=exit_id)

    if warehouse_exit.status == 'Chưa xuất':
        warehouse_exit.delete()
        messages.success(request, "Đã xóa phiếu xuất kho.")
    else:
        messages.error(request, "Không thể xóa phiếu xuất kho khi trạng thái là Đã xuất hoặc Đang xử lý.")
    
    return redirect('warehouse_exit_list')

from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseForbidden
from django.shortcuts import render, redirect

# View đăng nhập
def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        role = request.POST['role']  # Lấy vai trò từ form
        
        # Xác thực người dùng
        user = authenticate(request, username=username, password=password)
        
        if user:
            if role == 'admin' and not user.is_superuser:
                messages.error(request, "Tài khoản này không có quyền Admin.")
                return redirect('login')
            
            if role == 'nhanvien' and not user.is_staff:
                messages.error(request, "Tài khoản này không có quyền Nhân viên.")
                return redirect('login')
            
            # Đăng nhập người dùng
            login(request, user)
            
            # Điều hướng theo quyền người dùng
            if role == 'admin':
                return redirect('trangchu')  # Chuyển đến trang admin
            elif role == 'nhanvien':
                return redirect('warehouse_dashboard')  # Chuyển đến trang nhân viên
        else:
            messages.error(request, 'Tài khoản hoặc mật khẩu không đúng.')
    
    return render(request, 'csvapp/login.html')

# View trang chủ (Admin)
@login_required
def trangchu(request):  # Admin được phép truy cập
    return render(request, 'csvapp/trangchu.html', {'role': 'Admin'})

# View quản lý kho (Nhân viên)
@login_required
def warehouse_dashboard_view(request):  
    # Nhân viên hoặc Admin đều được phép truy cập
    if not request.user.is_staff:
        return HttpResponseForbidden("Bạn không có quyền truy cập vào trang này!")
    return render(request, 'csvapp/warehouse_dashboard.html', {'role': 'Nhân viên'})

# View đăng xuất
@login_required
def logout_view(request):
    logout(request)
    return redirect('login')
