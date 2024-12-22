# site1/urls.py
from django.contrib import admin
from django.urls import path
from csvapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path("warehouse_dashboard/", views.warehouse_dashboard_view, name="warehouse_dashboard"),
    path('', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('trangchu/', views.trangchu, name='trangchu'),
    path('upload/', views.upload_files, name='upload_files'),
    path('get-supplier-products/<int:supplier_id>/', views.get_supplier_products, name='get_supplier_products'),
    path('success/', views.success, name='success'),
    path('action1/', views.action1, name='action1'),
    path('action2/', views.action2, name='action2'),
    path('action3/', views.action3, name='action3'),
    path('action4/', views.action4, name='action4'),
    path('action5/', views.action5, name='action5'),
    path('search_orders_view/', views.search_orders_view, name='search_orders_view'),
    path('search_orders_date/', views.search_orders_date, name='search_orders_date'),
    path('suggest_products/', views.suggest_products, name='suggest_products'),
  
    path('edit_product/<int:product_id>/', views.edit_product, name='edit_product'),
    path('products/', views.product_management, name='product_management'),
    path('products/add/', views.add_product, name='add_product'),
    path('products/edit/<int:product_id>/', views.edit_product, name='edit_product'),
    path('products/delete/<int:product_id>/', views.delete_product, name='delete_product'),
    path('update_status/<int:entry_id>/', views.update_status, name='update_status'),
    path('locations/', views.location_management, name='location_management'),
    path('locations/add/', views.add_location, name='add_location'),
    path('locations/edit/<int:location_id>/', views.edit_location, name='edit_location'),
    path('locations/delete/<int:location_id>/', views.delete_location, name='delete_location'),
    path('warehouse/exit/update_status/<int:exit_id>/', views.update_exit_status, name='update_exit_status'),
    path('suppliers/', views.supplier_management, name='supplier_management'),
    path('suppliers/add/', views.add_supplier, name='add_supplier'),
    path('suppliers/edit/<int:supplier_id>/', views.edit_supplier, name='edit_supplier'),
    path('suppliers/delete/<int:supplier_id>/', views.delete_supplier, name='delete_supplier'),
    path('get-supplier-products-for-exit/<int:supplier_id>/', views.get_supplier_products, name='get_supplier_products'),
    path('products/', views.product_list, name='product_list'),
     path('product/<int:id>/', views.product_detail, name='product_detail'),
    path('products/<int:product_id>/add_image/', views.add_product_image, name='add_product_image'),
    path('products/<int:product_id>/delete_image/<int:image_id>/', views.delete_product_image, name='delete_product_image'),

    path('warranties/', views.warranty_list, name='warranty_list'),
    path('warranties/add/', views.add_warranty, name='add_warranty'),
    path('warranties/edit/<int:warranty_id>/', views.edit_warranty, name='edit_warranty'),
    path('warranties/delete/<int:warranty_id>/',views.delete_warranty, name='delete_warranty'),
    path('warehouse-entry/', views.warehouse_entry, name='warehouse_entry'),
    path('warehouse/entry/list/', views.warehouse_entry_list, name='warehouse_entry_list'),
    path('warehouse-entry/edit/<int:entry_id>/', views.edit_warehouse_entry, name='edit_warehouse_entry'),
    path('warehouse-entry/delete/<int:entry_id>/', views.delete_warehouse_entry, name='delete_warehouse_entry'), 
    path('warehouse-exit/', views.warehouse_exit, name='warehouse_exit'),
    path('warehouse/exit/list/', views.warehouse_exit_list, name='warehouse_exit_list'),
    path('warehouse-exit/edit/<int:exit_id>/', views.warehouse_exit_edit, name='warehouse_exit_edit'),
    path('warehouse-exit/delete/<int:exit_id>/', views.warehouse_exit_delete, name='warehouse_exit_delete'),
]   

from django.conf import settings
from django.conf.urls.static import static
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

