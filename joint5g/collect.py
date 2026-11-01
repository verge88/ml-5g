"""Run the existing benign native collector under the registered deployment IDs."""
import asyncio
from fullsuite.native_hook import run
if __name__=='__main__':asyncio.run(run())
