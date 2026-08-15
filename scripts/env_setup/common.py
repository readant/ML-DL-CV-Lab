import sys


def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_item(name, status, details=""):
    icon = "✅" if status else "❌"
    if details:
        print(f"  {icon} {name}: {details}")
    else:
        print(f"  {icon} {name}")
