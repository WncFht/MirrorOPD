# Reuse the repository's class without treating its path as the class name.
$ENV{'TEXINPUTS'} = '../book//:' . ($ENV{'TEXINPUTS'} // '');
