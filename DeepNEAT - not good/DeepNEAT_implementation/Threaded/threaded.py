from __future__ import print_function

import warnings

try:
    import threading
except ImportError:
    import dummy_threading as threading
    HAVE_THREADS = False
else:
    HAVE_THREADS = True

try:
    import Queue as queue
except ImportError:
    import queue

class ThreadedEvaluator(object):
    def __init__(self, num_workers, eval_function):
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.workers = []
        self.working = False
        self.inqueue = queue.Queue()
        self.outqueue = queue.Queue()

        if not HAVE_THREADS:
            warnings.warn("No threads available; use ParallelEvaluator, not ThreadedEvaluator")

    def __del__(self):
        if self.working:
            self.stop()

    def start(self):
        if self.working:
            return
        self.working = True
        for i in range(self.num_workers):
            w = threading.Thread(
                name="Worker Thread #{i}".format(i=i),
                target=self._worker,
            )
            w.daemon = True
            w.start()
            self.workers.append(w)

    def stop(self):
        self.working = False
        for w in self.workers:
            w.join()
        self.workers = []

    def _worker(self):
        while self.working:
            try:
                genome_id, genome, config = self.inqueue.get(
                    block=True,
                    timeout=0.2,
                )
            except queue.Empty:
                continue
            f = self.eval_function(genome, config)
            self.outqueue.put((genome_id, genome, f))

    def evaluate(self, genomes, config):
        if not self.working:
            self.start()
        p = 0
        for genome_id, genome in genomes:
            p += 1
            self.inqueue.put((genome_id, genome, config))

        while p > 0:
            p -= 1
            ignored_genome_id, genome, fitness = self.outqueue.get()
            genome.fitness = fitness