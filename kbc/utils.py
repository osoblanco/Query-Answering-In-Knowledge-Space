import enum

class QuerDAG(enum.Enum):
    TYPE1_1 = "1_1"
    TYPE1_2 = "1_2"
    TYPE2_2 = "2_2"
    TYPE1_3 = "1_3"
    TYPE2_3 = "2_3"
    TYPE3_3 = "3_3"
    TYPE4_3 = "4_3"
    TYPE1_3_joint = '1_3_joint'


class DynKBCSingleton:
    __instance = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if DynKBCSingleton.__instance == None:
            DynKBCSingleton()
        return DynKBCSingleton.__instance


    def set_attr(self, kbc, chains, parts, target_ids, lhs_norm):
        self.kbc = kbc
        self.chains = chains
        self.parts = parts
        self.target_ids = target_ids
        self.lhs_norm = lhs_norm
        self.__instance = self

    def __init__(self,kbc = None, chains = None , parts = None, \
    target_ids = None, lhs_norm = None):
        """ Virtually private constructor. """
        if DynKBCSingleton.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            DynKBCSingleton.kbc = kbc
            DynKBCSingleton.chains = chains
            DynKBCSingleton.parts = parts
            DynKBCSingleton.target_ids = target_ids
            DynKBCSingleton.lhs_norm = lhs_norm
            DynKBCSingleton.__instance = self
