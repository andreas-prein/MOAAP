# # Add this near the top of your file, after other imports
# ENABLE_PROFILING = os.getenv('MEMORY_PROFILING', 'false').lower() == 'true'

# if ENABLE_PROFILING:
#     from memory_profiler import profile
# else:
#     def profile(func):
#         """No-op decorator when profiling is disabled"""
#         return func

import functools
import time


def profile_sections(func):
    """
    Decorator to profile execution time of code sections within a function.\n
    Usage:\n
    @profile_sections\n
    def my_function(..., _timer=None):\n
        with _timer("section1"):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        section_times = {}
        
        def time_section(name):
            return SectionTimer(name, section_times)
        
        # Inject timer into function
        kwargs['_timer'] = time_section
        result = func(*args, **kwargs)
        
        # Print results
        total_time = sum(section_times.values())
        print(f"\n{func.__name__} timing breakdown:")
        for name, duration in sorted(section_times.items(), key=lambda x: x[1], reverse=True):
            percentage = (duration / total_time) * 100 if total_time > 0 else 0
            print(f"  {name}: {duration:.3f}s ({percentage:.1f}%)")
        
        return result
    return wrapper

class SectionTimer:
    def __init__(self, name, results_dict):
        self.name = name
        self.results_dict = results_dict
        
    def __enter__(self):
        self.start = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        self.results_dict[self.name] = time.perf_counter() - self.start



def timer(start,end):
    """
    Prints elapsed time in hh:mm:ss.ss format.
    
    Parameters
    ----------
    start : float
        Start time in seconds.
    end : float
        End time in seconds.
        
    Returns
    -------
    None (output via print)
    """
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("        "+"{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))