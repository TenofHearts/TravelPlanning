class Planner:
    def __init__(self, llm_model, prompt, notebook_ptr,
                 success_obj):
        self.llm = llm_model
        self.ans = ""
        self.prompt = prompt
        self.notebook_ptr = notebook_ptr
        self.success_obj = success_obj

    def __call__(self, query):
        print("Note:")
        print(self.notebook_ptr.read())
        self.task_status = True
        prompt = self.prompt + self.notebook_ptr.read() + query
        # self.ans = self.llm(self.prompt + self.notebook_ptr.read() + query)
        self.ans = self.llm([{"role": "user", "content": prompt}])
        return self.success_obj

    def get_ans(self):
        return self.ans

    def reset(self):
        self.ans = ""
