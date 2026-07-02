import sys
from os import path

def generate_html(output_dir, size):
    sys.stderr.write('Generating HTML to send...\n')

    html_path = path.join(output_dir, 'big_file.html')

    # check if index.html already exists
    if path.isfile(html_path) and path.getsize(html_path) > size:
        sys.stderr.write('index.html already exists\n')
        return

    head_text = ('HTTP/1.1 200 OK\n'
                 'X-Original-Url: https://www.example.org/quic-data/www.example.org/big_file.html\n'
                 '\n'
                 '<!DOCTYPE html>\n'
                 '<html>\n'
                 '<body>\n'
                 '<p>\n')

    foot_text = ('</p>\n'
                 '</body>\n'
                 '</html>\n')

    html = open(html_path, 'w')
    html.write(head_text)

    block_size = 100 * 1024 * 1024
    block = 'x' * block_size
    num_blocks = int(size) / block_size + 1
    for _ in xrange(num_blocks):
        html.write(block + '\n')

    html.write(foot_text)
    html.close()

if __name__ == '__main__':
    output_dir = "/home/cristiano/Desktop"
    size = 9 * 10**8
    generate_html(output_dir, size)