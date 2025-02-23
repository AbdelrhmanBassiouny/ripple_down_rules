import subprocess

def lint_code(code):
    with open("temp_script.py", "w") as f:
        f.write(code)

    result = subprocess.run(['pylint', 'temp_script.py'], capture_output=True, text=True)
    print(result.stdout)


class A:
    a: int = 1

class B:
    b: int = 2


while True:
    user_code = input("Write Python code >>> ")
    if user_code.strip().lower() in ["exit", "quit"]:
        break
    try:
        exec(user_code)
    except Exception as e:
        print(f"Error: {e}")

print(a.a)
