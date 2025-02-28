# Simple LL(1) Parser Implementation for beginners
# Grammar:
# A -> da | BC
# B -> g | Є
# C -> h | Є

# First, let's define a simple way to represent our grammar
grammar = {
    'A': [['d', 'a'], ['B', 'C']],
    'B': [['g'], ['Є']],  # Epsilon is represented as 'Є'
    'C': [['h'], ['Є']]
}

# Let's define our constants
EPSILON = 'Є'
EOF = '$'

def compute_first(grammar):
    """
    Compute FIRST sets for all symbols in the grammar
    FIRST(X) = set of terminals that can appear as the first symbol of any string
    derived from X
    """
    # Initialize FIRST sets for all non-terminals
    first = {}
    for non_terminal in grammar:
        first[non_terminal] = set()
    
    # Continue until no changes are made to any FIRST set
    while True:
        updated = False
        
        for non_terminal in grammar:
            for production in grammar[non_terminal]:
                # Case 1: If X -> Є is a production, add Є to FIRST(X)
                if production[0] == EPSILON:
                    if EPSILON not in first[non_terminal]:
                        first[non_terminal].add(EPSILON)
                        updated = True
                    continue
                
                # Case 2: If X -> Y... is a production
                current_symbol = production[0]
                
                # If the first symbol is a terminal, add it to FIRST(X)
                if current_symbol not in grammar:  # It's a terminal
                    if current_symbol not in first[non_terminal]:
                        first[non_terminal].add(current_symbol)
                        updated = True
                else:  # It's a non-terminal
                    # We need to compute FIRST for this non-terminal first
                    # Add all elements from FIRST(current_symbol) except Є to FIRST(X)
                    for terminal in first[current_symbol]:
                        if terminal != EPSILON and terminal not in first[non_terminal]:
                            first[non_terminal].add(terminal)
                            updated = True
                    
                    # If FIRST(current_symbol) contains Є, we need to consider the next symbol
                    # This is a simplified approach - for a full implementation, we would need
                    # to check if all symbols in the production can derive Є
                    if EPSILON in first[current_symbol] and len(production) > 1:
                        # Simplified: just add the first symbol of the next part if it's a terminal
                        next_symbol = production[1]
                        if next_symbol not in grammar:  # It's a terminal
                            if next_symbol not in first[non_terminal]:
                                first[non_terminal].add(next_symbol)
                                updated = True
        
        # If no updates were made in this iteration, we're done
        if not updated:
            break
    
    return first

def compute_follow(grammar, first):
    """
    Compute FOLLOW sets for all non-terminals in the grammar
    FOLLOW(A) = set of terminals that can appear immediately after A in some sentential form
    """
    # Initialize FOLLOW sets for all non-terminals
    follow = {}
    for non_terminal in grammar:
        follow[non_terminal] = set()
    
    # Add $ to FOLLOW of the start symbol (assuming 'A' is the start symbol)
    follow['A'].add(EOF)
    
    # Continue until no changes are made to any FOLLOW set
    while True:
        updated = False
        
        for non_terminal in grammar:
            for production_nt in grammar:
                for production in grammar[production_nt]:
                    # Skip epsilon productions
                    if production[0] == EPSILON:
                        continue
                    
                    # Look for non_terminal in the production
                    for i in range(len(production)):
                        if production[i] == non_terminal:
                            # Case 1: If A -> αBβ, add FIRST(β) - {Є} to FOLLOW(B)
                            if i < len(production) - 1:
                                next_symbol = production[i + 1]
                                
                                # If next symbol is a terminal, add it to FOLLOW(non_terminal)
                                if next_symbol not in grammar:
                                    if next_symbol not in follow[non_terminal]:
                                        follow[non_terminal].add(next_symbol)
                                        updated = True
                                else:  # next symbol is a non-terminal
                                    # Add FIRST(next_symbol) - {Є} to FOLLOW(non_terminal)
                                    for terminal in first[next_symbol]:
                                        if terminal != EPSILON and terminal not in follow[non_terminal]:
                                            follow[non_terminal].add(terminal)
                                            updated = True
                                    
                                    # Case 2: If A -> αBβ and Є is in FIRST(β), add FOLLOW(A) to FOLLOW(B)
                                    if EPSILON in first[next_symbol]:
                                        for terminal in follow[production_nt]:
                                            if terminal not in follow[non_terminal]:
                                                follow[non_terminal].add(terminal)
                                                updated = True
                            
                            # Case 3: If A -> αB, add FOLLOW(A) to FOLLOW(B)
                            elif i == len(production) - 1:
                                for terminal in follow[production_nt]:
                                    if terminal not in follow[non_terminal]:
                                        follow[non_terminal].add(terminal)
                                        updated = True
        
        # If no updates were made in this iteration, we're done
        if not updated:
            break
    
    return follow

def create_parse_table(grammar, first, follow):
    """
    Create an LL(1) parse table for the given grammar
    """
    parse_table = {}
    
    # Initialize parse table with empty dictionaries
    for non_terminal in grammar:
        parse_table[non_terminal] = {}
    
    # Fill in the parse table
    for non_terminal in grammar:
        for i, production in enumerate(grammar[non_terminal]):
            # Case 1: For each terminal a in FIRST(α), add A -> α to M[A, a]
            first_of_production = get_first_of_string(production, first, grammar)
            
            for terminal in first_of_production:
                if terminal != EPSILON:
                    if terminal in parse_table[non_terminal]:
                        print(f"Grammar is not LL(1): Conflict at [{non_terminal}, {terminal}]")
                    else:
                        parse_table[non_terminal][terminal] = production
            
            # Case 2: If Є is in FIRST(α), add A -> α to M[A, b] for each b in FOLLOW(A)
            if EPSILON in first_of_production:
                for terminal in follow[non_terminal]:
                    if terminal in parse_table[non_terminal]:
                        print(f"Grammar is not LL(1): Conflict at [{non_terminal}, {terminal}]")
                    else:
                        parse_table[non_terminal][terminal] = production
    
    return parse_table

def get_first_of_string(string, first, grammar):
    """
    Compute FIRST set for a string of grammar symbols
    """
    # If string is empty, return {Є}
    if not string or string[0] == EPSILON:
        return {EPSILON}
    
    result = set()
    
    # If the first symbol is a terminal, return {first symbol}
    if string[0] not in grammar:
        return {string[0]}
    
    # If the first symbol is a non-terminal
    # Add all elements from FIRST(first symbol) except Є to result
    for terminal in first[string[0]]:
        if terminal != EPSILON:
            result.add(terminal)
    
    # If Є is in FIRST(first symbol) and string has more symbols
    if EPSILON in first[string[0]] and len(string) > 1:
        # Recursively compute FIRST for the rest of the string
        rest_first = get_first_of_string(string[1:], first, grammar)
        for terminal in rest_first:
            result.add(terminal)
    
    # If all symbols in string can derive Є, add Є to result
    all_derive_epsilon = True
    for symbol in string:
        if symbol not in grammar:  # It's a terminal
            all_derive_epsilon = False
            break
        if EPSILON not in first[symbol]:
            all_derive_epsilon = False
            break
    
    if all_derive_epsilon:
        result.add(EPSILON)
    
    return result

def parse_input(input_string, grammar, parse_table):
    """
    Parse an input string using the LL(1) parsing algorithm
    """
    # Initialize the stack with EOF and the start symbol
    stack = [EOF, 'A']  # 'A' is our start symbol
    
    # Add EOF to the end of input
    input_string = input_string + EOF
    input_pos = 0
    
    # Parsing steps
    print("\nParsing Steps:")
    print("Stack           Input          Action")
    
    while stack:
        # Get the top of the stack
        top = stack[-1]
        
        # Get the current input symbol
        current_input = input_string[input_pos] if input_pos < len(input_string) else EOF
        
        # Adjusted spacing for better alignment
        stack_str = str(stack)
        input_str = input_string[input_pos:]
        print(f"{stack_str:<20}{input_str:<20}", end="")
        
        # If top is EOF and current input is EOF, we're done
        if top == EOF and current_input == EOF:
            print("Accept!")
            return True
        
        # If top is a terminal or EOF
        if top not in grammar:
            if top == current_input:
                # Match! Pop the stack and advance input
                stack.pop()
                input_pos += 1
                print(f"Match {top}")
            else:
                # Error: Expected top but got current_input
                print(f"Error: Expected {top}, got {current_input}")
                return False
        else:  # top is a non-terminal
            # Check if there's an entry in the parse table
            if current_input in parse_table[top]:
                # Get the production
                production = parse_table[top][current_input]
                
                # Pop the non-terminal
                stack.pop()
                
                # Push the production in reverse order (except Є)
                if production[0] != EPSILON:
                    for symbol in reversed(production):
                        stack.append(symbol)
                
                print(f"Apply {top} -> {production}")
            else:
                # Error: No production for [top, current_input]
                print(f"Error: No production for [{top}, {current_input}]")
                return False
    
    return True

# Main function to test our parser
def main():
    print("Grammar:")
    for nt in grammar:
        for prod in grammar[nt]:
            print(f"{nt} -> {''.join(prod)}")
    
    print("\nComputing FIRST sets...")
    first = compute_first(grammar)
    print("FIRST sets:")
    for nt in grammar:
        print(f"FIRST({nt}) = {first[nt]}")
    
    print("\nComputing FOLLOW sets...")
    follow = compute_follow(grammar, first)
    print("FOLLOW sets:")
    for nt in grammar:
        print(f"FOLLOW({nt}) = {follow[nt]}")
    
    print("\nCreating parse table...")
    parse_table = create_parse_table(grammar, first, follow)
    print("Parse Table:")
    for nt in grammar:
        for terminal in parse_table[nt]:
            print(f"[{nt}, {terminal}] = {nt} -> {''.join(parse_table[nt][terminal])}")
    
    # Test parsing
    test_input = "da"
    print(f"\nParsing input: {test_input}")
    result = parse_input(test_input, grammar, parse_table)
    if result:
        print("\nParsing successful!")
    else:
        print("\nParsing failed!")

if __name__ == "__main__":
    main()