FROM e2e-witness-root
# Import the original handler without starting its original blocking server.
RUN python3 -c "from pathlib import Path;p=Path('/guest.py');s=p.read_text();s=s.replace(\"HTTPServer(('0.0.0.0',8081),Handler).serve_forever()\",'');Path('/e2e5g_guest.py').write_text(s)"
COPY diagnose5g/guest.py /guest.py
