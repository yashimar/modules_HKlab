EXIT_STATE= '__exit__'
ORIGIN_STATE= '__origin__'

def Print(*s):
    for ss in s:
        print(ss)
    print("")
    
class TAsciiColors:
    Header  = '\033[95m'
    OKBlue  = '\033[94m'
    OKGreen = '\033[92m'
    Warning = '\033[93m'
    Fail    = '\033[91m'
    EndC    = '\033[0m'
    
def DPrint(*s):
    first = True
    for ss in s:
        if not first:
            print(ss)
        else:
            print(TAsciiColors.OKGreen+str(ss))
    print(TAsciiColors.EndC+"")
    
class TFSMConditionAction:    
    def __init__(self):
        self.Condition = lambda: False
        self.Action = None
        self.NextState = ""

class TFSMState:    
    def __init__(self):
        self.EntryAction = None
        self.ExitAction = None
        self.Actions = []
        self.ElseAction = TFSMConditionAction()
    
    def NewAction(self):
        self.Actions.append(TFSMConditionAction())
        return self.Actions[-1]
    
class TStateMachine:
    def __init__(self, start_state=""):
        self.States = {}
        self.StartState = start_state
        self.Debug = False
        
    def __getitem__(self, key):
        return self.States[key]
    
    def __setitem__(self, key, value):
        self.States[key] = value
    
    def Show(self):
        for id, st in self.StartState.items():
            print(f"[{id}].EntryAction = {st.EntryAction}")
            print(f"[{id}].ExitAction = {st.ExitAction}")
            print(f"[{id}].ElseAction = {st.ElseAction}")
            a_id = 0
            for a in st.Actions:
                print(f"[{id}].Actions[{a_id}].Condition = {a.Condition}")
                print(f"[{id}].Actions[{a_id}].Action = {a.Action}")
                print(f"[{id}].Actions[{a_id}].NextState = {a.NextState}")
                a_id += 1
            print("")
        print("StartState = {self.StartState}")
        print("Debug = {self.Debug}")
        
    def SetStartState(self, start_state=""):
        self.StartState = start_state
        
    def Run(self):
        self.prev_state = ""
        self.curr_state = self.StartState
        count = 0
        while self.curr_state!="":
            count += 1
            if self.Debug: DPrint("@", count, curr_state)
            st = self.States[self.curr_state]
            if st.EntryAction and self.prev_state!=self.curr_state:
                if self.Debug: DPrint("@", count, self.curr_state, "EntryAction")
                st.EntryAction()
                
            a_id = 0
            a_id_satisfied = -1
            next_state = ""
            for ca in st.Actions:
                if ca.Condition():
                    if a_id_satisfied>=0:
                        print(f"Warning: multiple conditions are satisfied in {self.curr_state}")
                        print(f"  First satisfied condition index & next state: {a_id_satisfied}, {next_state}")
                        print(f"  Additionally satisfied condition index & next state: {a_id}, {ca.NextState}")
                        print("  First conditioned action is activated")
                    else:
                        a_id_satisfied = a_id
                        next_state = ca.NextState
                a_id += 1
                
            if a_id_satisfied>=0:
                if self.Debug: DPrint("@", count, self.curr_state, "Condition satisfied:", a_id_satisfied)
                if st.Actions[a_id_satisfied].Action:
                    if self.Debug: DPrint("@", count, self.curr_state, "Action", a_id_satisfied)
                    st.Actions[a_id_satisfied].Action()
            else:
                if st.ElseAction.Condition():
                    if st.ElseAction.Action:
                        if self.Debug: DPrint("@", count, self.curr_state, "ElseAction")
                        st.ElseAction.Action()
                    next_state = st.ElseAction.NextState
                    
            if self.Debug: DPrint("@", count, self.curr_state, "Next state:", next_state)
            
            if next_state=="ORIGIN_STATE" or next_state==self.curr_state:
                self.prev_state = self.curr_state
            else:
                if st.ExitAction:
                    if self.Debug: DPrint('@',count,self.curr_state,'ExitAction')
                    st.ExitAction()
                if next_state=="":
                    print(f"Next state is not defined at {self.curr_state}. Hint: use ElseAction to specify the case where no conditions are satisfied.")
                self.prev_state = self.curr_state
                if next_state==EXIT_STATE:
                    self.curr_state = ""
                else:
                    self.curr_state = next_state