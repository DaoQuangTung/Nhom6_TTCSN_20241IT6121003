from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .models import Product, WarehouseEntry, WarehouseExit, Warranty, Supplier, Location

# User registration form
class UserRegisterForm(UserCreationForm):
    class Meta:
        model = User
        fields = ('username', 'password1', 'password2')

# Product form
class ProductForm(forms.ModelForm):
    class Meta:
        model = Product
        fields = ['name', 'description', 'price', 'quantity', 'supplier', 'location']

from django import forms
from .models import WarehouseEntry, Product, EntryProduct, Warranty, Supplier, Location

# forms.py
from django import forms
from .models import WarehouseEntry, Supplier

class WarehouseEntryForm(forms.ModelForm):
    class Meta:
        model = WarehouseEntry
        fields = ['entry_date', 'supplier']


class EntryProductForm(forms.ModelForm):
    class Meta:
        model = EntryProduct
        fields = ['product', 'quantity']





# Supplier form
from django import forms
from django.core.validators import RegexValidator
from .models import Supplier

class SupplierForm(forms.ModelForm):
    phone_validator = RegexValidator(
        regex=r'^\d{10}$',  # Chỉ cho phép 10 chữ số
        message='Số điện thoại phải có 10 chữ số.'
    )

    phone = forms.CharField(
        max_length=20,
        validators=[phone_validator],  # Áp dụng validator cho trường phone
        error_messages={
            'required': 'Trường này là bắt buộc.',
            'max_length': 'Số điện thoại không được quá 20 ký tự.'
        }
    )

    class Meta:
        model = Supplier
        fields = ['name', 'address', 'phone', 'email']


# Warranty form
class WarrantyForm(forms.ModelForm):
    class Meta:
        model = Warranty
        fields = ['product', 'warranty_period', 'warranty_type']

# Location form
class LocationForm(forms.ModelForm):
    class Meta:
        model = Location
        fields = ['name', 'description']
from django import forms
from .models import ProductImage

class ProductImageForm(forms.ModelForm):
    class Meta:
        model = ProductImage
        fields = ['image']  # or the fields you want in the form

# forms.py
from django import forms
from .models import WarehouseExit, ExitProduct

from django import forms
from .models import WarehouseExit
from django import forms
from .models import WarehouseExit
# forms.py
class WarehouseExitForm(forms.ModelForm):
    class Meta:
        model = WarehouseExit
        fields = ['exit_date', 'supplier', 'status']  # Bao gồm trường 'status' và 'supplier'
        widgets = {
            'exit_date': forms.DateInput(attrs={'type': 'date'}),
            'status': forms.Select(choices=WarehouseExit.STATUS_CHOICES),
        }

    def __init__(self, *args, **kwargs):
        super(WarehouseExitForm, self).__init__(*args, **kwargs)
        self.fields['supplier'].queryset = Supplier.objects.all()  # Chỉ lấy các nhà cung cấp hiện có
class ExitProductForm(forms.ModelForm):
    class Meta:
        model = ExitProduct
        fields = ['product', 'quantity']

from django import forms
from django.contrib.auth.forms import AuthenticationForm
from .models import Profile
class CustomAuthenticationForm(AuthenticationForm):
    ROLE_CHOICES = [
        ('Admin', 'Admin'),
        ('Nhân viên', 'Nhân viên'),
    ]
    role = forms.ChoiceField(choices=ROLE_CHOICES, required=True, label="Vai trò")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['username'].widget.attrs.update({'placeholder': 'Tên đăng nhập'})
        self.fields['password'].widget.attrs.update({'placeholder': 'Mật khẩu'})

from django import forms
from django.contrib.auth.models import User

class RegistrationForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['username', 'password', 'email']

    password = forms.CharField(widget=forms.PasswordInput())

    def save(self, commit=True):
        user = super().save(commit=False)
        if commit:
            user.set_password(self.cleaned_data["password"])
            user.save()

            # Nếu người dùng mới được tạo, tạo Profile cho họ
            Profile.objects.create(user=user)  # Tạo Profile cho User mới
        return user
