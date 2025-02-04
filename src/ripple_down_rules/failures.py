

class InvalidOperator(Exception):
    def __init__(self, rule_str: str, valid_operators: list):
        self.rule_str = rule_str
        self.valid_operators = valid_operators

    def __str__(self):
        return f"Invalid operator in {self.rule_str}, valid operators are: {self.valid_operators}"

