import json


class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

    def display_info(self):
        print(f"Employee Name: {self.name}")
        print(f"Salary: ${self.salary}")

    def get_annual_salary(self):
        return self.salary * 12

    def get_name(self):
        return self.name

    def get_salary(self):
        return self.salary


try:
    with open("dataa.txt", "r") as f:
        data = f.read()
    print(data)
except FileNotFoundError:
    print("File not found. Please check the file path and try again.")

data = {"name": "Rahul"}

with open("user.json", "w") as f:
    json.dump(data, f)

employee1 = Employee("Alice", 5000)
employee1.display_info()
