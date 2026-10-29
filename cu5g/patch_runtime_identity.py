"""Inject Stage 38 NF-side runtime identity witness into Open5GS SBI context."""
from pathlib import Path

path = Path("/src/open5gs/lib/sbi/context.c")
text = path.read_text()

include_needle = '#include "ogs-sbi.h"\n'
include_insert = include_needle + '#include <stdio.h>\n#include <stdlib.h>\n#include <unistd.h>\n'
if "STAGE38_WITNESS_DIR" not in text:
    if include_needle not in text:
        raise SystemExit("context include marker not found")
    text = text.replace(include_needle, include_insert, 1)

needle = "    ogs_sbi_nf_instance_set_type(self.nf_instance, nf_type);\n"
instrument = needle + r'''
    {
        const char *stage38_dir = getenv("STAGE38_WITNESS_DIR");
        if (stage38_dir && *stage38_dir) {
            char witness_path[512];
            FILE *witness_fp = NULL;
            snprintf(witness_path, sizeof(witness_path),
                    "%s/%ld.tsv", stage38_dir, (long)getpid());
            witness_fp = fopen(witness_path, "w");
            if (witness_fp) {
                fprintf(witness_fp, "%ld\t%d\t%s\n",
                        (long)getpid(), (int)nf_type, nf_instance_id);
                fclose(witness_fp);
            } else {
                ogs_error("STAGE38: cannot write NF-side witness [%s]",
                        witness_path);
            }
        }
    }
'''
if "cannot write NF-side witness" not in text:
    if needle not in text:
        raise SystemExit("NF instance marker not found")
    text = text.replace(needle, instrument, 1)

path.write_text(text)
print("STAGE38_SBI_CONTEXT_PATCH_OK")
