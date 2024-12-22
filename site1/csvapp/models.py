from django.db import models
from django.core.validators import RegexValidator

class Supplier(models.Model):
    name = models.CharField(max_length=255)
    address = models.TextField()
    phone_validator = RegexValidator(regex=r'^\d{10}$', message='Số điện thoại phải có 10 chữ số.')
    phone = models.CharField(max_length=20, validators=[phone_validator])
    email = models.EmailField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class Location(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class Product(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(null=True, blank=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    quantity = models.IntegerField()
    supplier = models.ForeignKey(Supplier, on_delete=models.SET_NULL, null=True, related_name='products')
    location = models.ForeignKey(Location, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class WarehouseEntry(models.Model):
    PENDING = 'PENDING'
    PROCESSING = 'PROCESSING'
    COMPLETED = 'COMPLETED'
    
    STATUS_CHOICES = [
        (PENDING, 'Chưa nhập'),
        (PROCESSING, 'Đang xử lý'),
        (COMPLETED, 'Đã nhập'),
    ]
    
    entry_date = models.DateField()
    supplier = models.ForeignKey(Supplier, on_delete=models.SET_NULL, null=True)
    total_quantity = models.IntegerField(default=0)  # Tổng số lượng sẽ được tính lại
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default=PENDING)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Nhập kho {self.entry_date} - {self.supplier.name}"

    def update_inventory(self):
        """Cập nhật số lượng sản phẩm trong kho khi phiếu nhập có trạng thái 'Đã nhập'."""
        if self.status == WarehouseEntry.COMPLETED:
            for entry_product in self.entry_products.all():
                product = entry_product.product
                product.quantity += entry_product.quantity
                product.save()

    def update_total_quantity(self):
        """Cập nhật lại tổng số lượng của phiếu nhập kho."""
        total = sum(entry_product.quantity for entry_product in self.entry_products.all())
        self.total_quantity = total
        self.save()


class EntryProduct(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    warehouse_entry = models.ForeignKey(WarehouseEntry, on_delete=models.CASCADE, related_name='entry_products')
    quantity = models.IntegerField()

from django.db import models

# models.py
# models.py
class WarehouseExit(models.Model):
    PENDING = 'PENDING'
    PROCESSING = 'PROCESSING'
    COMPLETED = 'COMPLETED'

    STATUS_CHOICES = [
        (PENDING, 'Chờ xử lý'),
        (PROCESSING, 'Đang xử lý'),
        (COMPLETED, 'Đã xuất'),
    ]

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=PENDING,
    )
    supplier = models.ForeignKey(Supplier, on_delete=models.CASCADE)
    exit_date = models.DateField()
    total_quantity = models.PositiveIntegerField(default=0)

    def update_inventory(self):
        """Cập nhật số lượng sản phẩm trong kho khi phiếu xuất có trạng thái 'Đã xuất'."""
        if self.status == self.COMPLETED:
            for exit_product in self.exit_products.all():
                product = exit_product.product
                product.quantity -= exit_product.quantity  # Giảm số lượng sản phẩm khi xuất kho
                product.save()

    def update_total_quantity(self):
        """Cập nhật lại tổng số lượng của phiếu xuất kho."""
        total = sum(exit_product.quantity for exit_product in self.exit_products.all())
        self.total_quantity = total
        self.save()

class ExitProduct(models.Model):
    warehouse_exit = models.ForeignKey(WarehouseExit, related_name='exit_products', on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField()

    def __str__(self):
        return f"{self.quantity} x {self.product.name}"


# InventoryAudit (Kiểm kê hàng hóa)
class InventoryAudit(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    audit_date = models.DateField()
    quantity_counted = models.IntegerField()
    discrepancy = models.IntegerField()  # Difference between actual quantity and counted quantity
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Kiểm kê {self.product.name} ngày {self.audit_date}"

# Warranty (Bảo hành)
class Warranty(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    warranty_period = models.IntegerField(help_text="Thời gian bảo hành (tháng)")
    warranty_type = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Bảo hành cho {self.product.name}"

# WarrantyClaim (Yêu cầu bảo hành)
class WarrantyClaim(models.Model):
    warranty = models.ForeignKey(Warranty, on_delete=models.CASCADE)
    claim_date = models.DateField()
    reason = models.TextField()
    status = models.CharField(max_length=100, choices=[('Pending', 'Đang xử lý'), ('Approved', 'Đã chấp nhận'), ('Rejected', 'Bị từ chối')])
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Yêu cầu bảo hành ngày {self.claim_date} cho {self.warranty.product.name}"

class ProductImage(models.Model):
    product = models.ForeignKey(Product, related_name='images', on_delete=models.CASCADE)
    image = models.ImageField(upload_to='product_images/')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Image for {self.product.name}"
    
# csvapp/models.py
from django.db import models
from django.contrib.auth.models import User

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')  # Mối quan hệ một-một
    role = models.CharField(max_length=20, choices=[('Admin', 'Admin'), ('Nhân viên', 'Nhân viên')], default='Nhân viên')

    def __str__(self):
        return self.user.username
# models.py

from django.db import models

class Report(models.Model):
    file = models.FileField(upload_to='reports/')
    upload_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Report {self.id}"

class SaleReport(models.Model):
    report = models.ForeignKey(Report, on_delete=models.CASCADE, related_name='sales')
    order_id = models.CharField(max_length=255)
    product = models.CharField(max_length=255)
    quantity_ordered = models.IntegerField()
    price_each = models.DecimalField(max_digits=10, decimal_places=2)
    order_date = models.DateTimeField()
    purchase_address = models.CharField(max_length=255)

    def __str__(self):
        return f"SaleReport {self.id}"
