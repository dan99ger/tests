import os

def test_environment_variables():
    """اختبار بسيط للتأكد من وجود الملفات الأساسية"""
    assert os.path.exists("app.py"), "ملف app.py غير موجود!"