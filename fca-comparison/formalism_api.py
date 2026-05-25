"""
Python API mirroring the formalism's constructs.
LLMs generate code using these functions/classes instead of producing raw JSON.
The generated code builds the same data structures as the Pydantic schemas.
"""


# === Expressions ===

def Constant(value, unit=None):
    r = {"expr_type": "Constant", "value": value}
    if unit: r["unit"] = unit
    return r

def ArithOp(operator, left, right):
    return {"expr_type": "ArithOp", "operator": operator, "left": left, "right": right}

def Add(left, right): return ArithOp("+", left, right)
def Sub(left, right): return ArithOp("-", left, right)
def Mul(left, right): return ArithOp("*", left, right)
def Div(left, right): return ArithOp("/", left, right)

def Negate(operand):
    return {"expr_type": "Negate", "operand": operand}

def Now():
    return {"expr_type": "Now"}

def Prev(time):
    return {"expr_type": "Prev", "time": time}

def Next(time):
    return {"expr_type": "Next", "time": time}

def RelTime(time, offset):
    return {"expr_type": "RelTime", "time": time, "offset": offset}

def Diff(t1, t2):
    return {"expr_type": "Diff", "t1": t1, "t2": t2}

def MkInterval(t1, t2, left_open=False, right_open=False):
    return {"expr_type": "MkInterval", "t1": t1, "t2": t2, "left_open": left_open, "right_open": right_open}

def SinceZero(time):
    return {"expr_type": "SinceZero", "time": time}

def Start(interval):
    return {"expr_type": "Start", "interval": interval}

def End(interval):
    return {"expr_type": "End", "interval": interval}

def Duration(interval):
    return {"expr_type": "Duration", "interval": interval}

def Addr(entity):
    return {"expr_type": "Addr", "entity": entity}

def Val(entity, time):
    return {"expr_type": "Val", "entity": entity, "time": time}

def ValBefore(entity, time):
    return {"expr_type": "ValBefore", "entity": entity, "time": time}

def EvtOccCount(entity, interval):
    return {"expr_type": "EvtOccCount", "entity": entity, "interval": interval}

def LastOcc(entity, time, n=1):
    return {"expr_type": "LastOcc", "entity": entity, "time": time, "n": n}

def FirstOcc(entity, time, n=1):
    return {"expr_type": "FirstOcc", "entity": entity, "time": time, "n": n}

def MaxVal(entity, interval):
    return {"expr_type": "MaxVal", "entity": entity, "interval": interval}

def MinVal(entity, interval):
    return {"expr_type": "MinVal", "entity": entity, "interval": interval}

def Size(set_expr):
    return {"expr_type": "Size", "set_expr": set_expr}

def Filter(set_expr, variable, predicate):
    return {"expr_type": "Filter", "set_expr": set_expr, "variable": variable, "predicate": predicate}


# === Predicates ===

def Cmp(left, operator, right):
    return {"pred_type": "Cmp", "left": left, "operator": operator, "right": right}

def Not(predicate):
    return {"pred_type": "Not", "predicate": predicate}

def AllOf(*predicates):
    return {"pred_type": "AllOf", "predicates": list(predicates)}

def AnyOf(*predicates):
    return {"pred_type": "AnyOf", "predicates": list(predicates)}

def Implies(antecedent, consequent):
    return {"pred_type": "Implies", "antecedent": antecedent, "consequent": consequent}

def ForAll(set_expr, variable, predicate):
    return {"pred_type": "ForAll", "set_expr": set_expr, "variable": variable, "predicate": predicate}

def Exists(set_expr, variable, predicate):
    return {"pred_type": "Exists", "set_expr": set_expr, "variable": variable, "predicate": predicate}

def Happening(entity, time):
    return {"pred_type": "Happening", "entity": entity, "time": time}

def HasHappened(entity, time):
    return {"pred_type": "HasHappened", "entity": entity, "time": time}

def Operation(operation, entity, args=None):
    return {"pred_type": "Operation", "operation": operation, "entity": entity, "args": args or []}


# === Temporal Constructs ===

def Always(predicate):
    return {"construct_type": "Always", "predicate": predicate}

def Eventually(predicate):
    return {"construct_type": "Eventually", "predicate": predicate}

def Initial(predicate):
    return {"construct_type": "Initial", "predicate": predicate}

def Causes(condition, effect):
    return {"construct_type": "Causes", "condition": condition, "effect": effect}

def CausesWithin(condition, effect, duration, duration_unit="s"):
    return {"construct_type": "CausesWithin", "condition": condition, "effect": effect, "duration": duration, "duration_unit": duration_unit}

def Sequence(*steps):
    return {"construct_type": "Sequence", "steps": list(steps)}

def Precedes(before, after):
    return {"construct_type": "Precedes", "before": before, "after": after}

def Excludes(p1, p2):
    return {"construct_type": "Excludes", "p1": p1, "p2": p2}

def Immediately(trigger, effect):
    return {"construct_type": "Immediately", "trigger": trigger, "effect": effect}

def TraceAllOf(*constructs):
    return {"construct_type": "TraceAllOf", "constructs": list(constructs)}

def TraceAnyOf(*constructs):
    return {"construct_type": "TraceAnyOf", "constructs": list(constructs)}


# === Entities ===

def Entity(name, type, modifiers=None, description=""):
    return {"name": name, "type": type, "modifiers": modifiers or [], "description": description}

def Modifier(key, value=None):
    return {"key": key, "value": value}


# === Requirement ===

def Requirement(flavour, entities, constraint):
    return {"flavour": flavour, "entities": entities, "constraint": constraint}
