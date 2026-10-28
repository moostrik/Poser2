"""The phone page's files hang together: the structure a browser would punish silently (it drops a rule after a
stray brace, refuses a script that declares a name twice, finds no element for an id) is checked here, with
nothing but the text of relay/web."""

import os
import re
import unittest

WEB = os.path.join(os.path.dirname(__file__), '..', 'web')


def read(name: str) -> str:
    with open(os.path.join(WEB, name), encoding='utf-8') as f:
        return f.read()


def blank_css_comments(css: str) -> str:
    """The comments as spaces, the line count kept."""
    return re.sub(r'/\*.*?\*/', lambda m: re.sub(r'[^\n]', ' ', m.group()), css, flags=re.S)


def blank_js(js: str) -> str:
    """The script with its strings, template literals, regex-free comments blanked (their braces and quotes
    would mislead a brace count), everything else kept in place; `${...}` inside a template stays code."""
    out: list[str] = []
    i, n = 0, len(js)
    mode: list[str] = ['code']           # a stack: code, ', ", `, line, block; `${` pushes code onto a template
    while i < n:
        c = js[i]
        top = mode[-1]
        if top == 'code':
            if c == '/' and js.startswith('//', i):
                mode.append('line'); out.append(' '); i += 1; continue
            if c == '/' and js.startswith('/*', i):
                mode.append('block'); out.append(' '); i += 1; continue
            if c in '\'"`':
                mode.append(c); out.append(c); i += 1; continue
            if c == '}' and len(mode) > 1 and mode[-2] == '`':
                mode.pop(); out.append(c); i += 1; continue       # the end of ${...}: back in the template
            out.append(c); i += 1; continue
        if top == 'line':
            if c == '\n': mode.pop(); out.append(c)
            else: out.append(' ')
            i += 1; continue
        if top == 'block':
            if js.startswith('*/', i): mode.pop(); out.append('  '); i += 2
            else: out.append(c if c == '\n' else ' '); i += 1
            continue
        # inside a string or template
        if c == '\\':
            out.append('  '); i += 2; continue
        if c == top:
            mode.pop(); out.append(c); i += 1; continue
        if top == '`' and js.startswith('${', i):
            mode.append('code'); out.append('${'); i += 2; continue
        out.append(c if c == '\n' else ' '); i += 1
    return ''.join(out)


class TestStylesheet(unittest.TestCase):
    def setUp(self):
        self.css = blank_css_comments(read('style.css'))

    def test_braces_balance_and_no_rule_is_dropped(self):
        # a `}` at the top level makes the browser read the next rule's selector as `} selector`, invalid, and
        # drop that rule; a keyframe step outside @keyframes is an invalid rule too (both happened)
        depth = 0
        for n, line in enumerate(self.css.split('\n'), 1):
            stripped = line.strip()
            self.assertFalse(depth == 0 and stripped.startswith('}'), f'line {n}: a stray }} at the top level')
            self.assertFalse(depth == 0 and re.match(r'(\d+%|from|to)\s*\{', stripped), f'line {n}: a keyframe step outside @keyframes')
            depth += line.count('{') - line.count('}')
            self.assertGreaterEqual(depth, 0, f'line {n}: more }} than {{')
        self.assertEqual(depth, 0, 'braces left open at the end')

    def test_every_id_selector_names_an_element(self):
        html = read('index.html')
        ids = set(re.findall(r'\bid="([^"]+)"', html))
        # the preludes: the text before each `{` since the previous `{`, `}` or `;`
        for prelude in re.findall(r'(?:^|[{};])([^{};]*)\{', self.css):
            for name in re.findall(r'#([A-Za-z][\w-]*)', prelude):
                self.assertIn(name, ids, f'#{name} in the stylesheet names no element of index.html')


class TestScript(unittest.TestCase):
    def setUp(self):
        self.js = read('app.js')
        self.code = blank_js(self.js)

    def test_every_element_the_script_looks_up_exists(self):
        ids = set(re.findall(r'\bid="([^"]+)"', read('index.html')))
        for name in re.findall(r"\$\('([^']+)'\)", self.js):
            self.assertIn(name, ids, f"$('{name}') finds no element in index.html")

    def test_no_name_is_declared_twice_in_one_scope(self):
        # a second `const`/`let`/`function` of a name in the same block is a syntax error for the whole script
        # (it happened: nothing on the page ran); the braces give the scope, closely enough
        scopes: list[set] = [set()]
        depth_names = re.compile(r'\b(const|let|function)\s+([A-Za-z_$][\w$]*)')
        pos = 0
        for m in re.finditer(r'[{}]|\b(?:const|let|function)\s+[A-Za-z_$][\w$]*', self.code):
            token = m.group()
            if token == '{':
                scopes.append(set())
            elif token == '}':
                if len(scopes) > 1: scopes.pop()
            else:
                kind, name = depth_names.match(token).groups()
                line = self.code.count('\n', 0, m.start()) + 1
                self.assertNotIn(name, scopes[-1], f'line {line}: {kind} {name} is declared twice in one scope')
                scopes[-1].add(name)
                if kind in ('const', 'let'):
                    # the other names of `const a = 1, b = 2`: up to the `;`, the names after each top-level comma
                    rest = self.code[m.end():self.code.find(';', m.end())]
                    for other in re.findall(r',\s*([A-Za-z_$][\w$]*)\s*=', re.sub(r'\([^()]*\)|\[[^\[\]]*\]', '', rest)):
                        self.assertNotIn(other, scopes[-1], f'line {line}: {kind} {other} is declared twice in one scope')
                        scopes[-1].add(other)
            pos = m.end()
        self.assertEqual(len(scopes), 1, 'braces left open at the end of the script')

    def test_the_script_is_one_strict_script(self):
        self.assertIn("'use strict';", self.js)


class TestMarkup(unittest.TestCase):
    def test_the_page_loads_its_files(self):
        html = read('index.html')
        self.assertIn('href="web/style.css"', html)
        self.assertIn('src="web/app.js"', html)
        for name in re.findall(r'href="web/([^"]+)"', html):
            self.assertTrue(os.path.exists(os.path.join(WEB, name)), f'web/{name} is linked but missing')


if __name__ == '__main__':
    unittest.main()
