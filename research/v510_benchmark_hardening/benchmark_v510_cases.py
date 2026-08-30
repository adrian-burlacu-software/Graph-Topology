"""V510 adversarial cases.

These cases deliberately exercise semantic boundaries rather than adding new
operators. The suite is small enough to inspect by hand and deterministic
so it can serve as a stable regression target for later learned components.
"""

V510_ADVERSARIAL_CASES = [
    # Reference + semantic normalization.
    {"name":"ref_color_red","turns":["the dog is red","is it red?"],"checks":{"must_contain":"red"},"focus":"reference/color"},
    {"name":"ref_color_blue","turns":["the dog is blue","is it blue?"],"checks":{"must_contain":"blue"},"focus":"reference/color"},
    {"name":"ref_color_green","turns":["the dog is green","is it green?"],"checks":{"must_contain":"green"},"focus":"reference/color"},
    {"name":"ref_size_small","turns":["the dog is small","is it small?"],"checks":{"must_contain":"small"},"focus":"reference/size"},
    {"name":"ref_size_big","turns":["the dog is big","is it big?"],"checks":{"must_contain":"big"},"focus":"reference/size"},
    {"name":"ref_shape_round","turns":["the ball is round","is it round?"],"checks":{"must_contain":"round"},"focus":"reference/shape"},
    {"name":"ref_shape_square","turns":["the box is square","is it square?"],"checks":{"must_contain":"square"},"focus":"reference/shape"},

    # Topic switching and pronoun preservation.
    {"name":"switch_dog","turns":["the universe is huge","the dog is red","is it red?"],"checks":{"must_contain":"red","must_not_contain":"huge"},"focus":"reference/topic"},
    {"name":"switch_cat","turns":["the dog is red","the cat is blue","is it blue?"],"checks":{"must_contain":"blue","must_not_contain":"red"},"focus":"reference/topic"},
    {"name":"switch_book","turns":["the dog is red","the book is blue","is it blue?"],"checks":{"must_contain":"blue"},"focus":"reference/topic"},

    # Counts.
    {"name":"count_one_dog","turns":["there is one dog","how many dogs are there?"],"checks":{"must_contain_one":["one","1"]},"focus":"count"},
    {"name":"count_three_dogs","turns":["there are three dogs","how many dogs are there?"],"checks":{"must_contain_one":["three","3"]},"focus":"count"},
    {"name":"count_four_cats","turns":["there are four cats","how many cats are there?"],"checks":{"must_contain_one":["four","4"]},"focus":"count"},
    {"name":"count_two_books","turns":["there are two books","how many books are there?"],"checks":{"must_contain_one":["two","2"]},"focus":"count"},
    {"name":"count_two_people","turns":["there are two people","how many people are there?"],"checks":{"must_contain_one":["two","2"]},"focus":"count"},

    # State updates.
    {"name":"state_red_blue","turns":["the dog is red","the dog is blue","what color is the dog?"],"checks":{"must_contain":"blue"},"focus":"state"},
    {"name":"state_small_big","turns":["the dog is small","the dog is big","what size is the dog?"],"checks":{"must_contain":"big"},"focus":"state"},
    {"name":"state_color_three","turns":["the dog is red","the dog is green","the dog is blue","what color is the dog?"],"checks":{"must_contain":"blue"},"focus":"state"},

    # Deterministic operators.
    {"name":"arith_add","turns":["what is 7 plus 5?"],"checks":{"must_contain":"12"},"focus":"logic"},
    {"name":"arith_minus","turns":["what is 10 minus 3?"],"checks":{"must_contain":"7"},"focus":"logic"},
    {"name":"arith_times","turns":["what is 4 times 3?"],"checks":{"must_contain":"12"},"focus":"logic"},
    {"name":"arith_divide","turns":["what is 12 divided by 3?"],"checks":{"must_contain":"4"},"focus":"logic"},
    {"name":"letter_a","turns":["how many a's are in banana?"],"checks":{"must_contain":"3"},"focus":"logic"},
    {"name":"letter_b","turns":["how many b's are in bubble?"],"checks":{"must_contain":"3"},"focus":"logic"},
    {"name":"spell_banana","turns":["how do you spell banana?"],"checks":{"must_contain":"banana"},"focus":"logic"},

    # State arithmetic must not mutate state.
    {"name":"add_state","turns":["there are two dogs","what if we add 5 dogs?"],"checks":{"must_contain_one":["seven","7"]},"focus":"state-arithmetic"},
    {"name":"subtract_state","turns":["there are ten dogs","what if three leave?"],"checks":{"must_contain_one":["seven","7"]},"focus":"state-arithmetic"},
    {"name":"add_state_twice_no_mutation","turns":["there are two dogs","what if we add 5 dogs?","how many dogs are there?"],"checks":{"must_contain_one":["two","2"]},"focus":"state-arithmetic"},

    # Unknowns: architecture should not manufacture a factual property.
    {"name":"unknown_weight","turns":["there is a dog","what is the dog's weight?"],"checks":{"nonempty":True},"focus":"unknown"},
    {"name":"unknown_color","turns":["there is a dog","what color is the dog?"],"checks":{"nonempty":True},"focus":"unknown"},
    {"name":"unknown_shape","turns":["there is a dog","what shape is the dog?"],"checks":{"nonempty":True},"focus":"unknown"},

    # Scope / mixed entities.
    {"name":"mixed_animals","turns":["there is one cat and two dogs","how many animals are there?"],"checks":{"nonempty":True},"focus":"scope"},
    {"name":"mixed_dogs","turns":["there is one cat and two dogs","how many dogs are there?"],"checks":{"must_contain_one":["two","2"]},"focus":"scope"},
    {"name":"mixed_cats","turns":["there is one cat and two dogs","how many cats are there?"],"checks":{"must_contain_one":["one","1"]},"focus":"scope"},
]
