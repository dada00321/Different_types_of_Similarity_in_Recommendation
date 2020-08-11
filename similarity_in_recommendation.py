import numpy as np

class Similarity():
    def exception_handle(self, vectors):
        dim = len(vectors[0]) # dim: dimension
        if not any([len(vectors) != 2, dim != len(vectors[1])]):
            return False, dim
        else: # if any error happens, return true
            if len(vectors) != 2:
                print("The number of points in space can only be 2!")
            elif dim != len(vectors[1]):
                print("Dimensions of two vectors are not be same.")
            return True, None
        
    # calculating the distance of 2 points in n-dimensional space
    def Euclidean_distance(self, *vectors):
        err_state, dim = self.exception_handle(vectors)
        if err_state == False:
            return np.sqrt(sum([(vectors[0][d]-vectors[1][d])**2 for d in range(dim)]))
    
    def Manhattan_distance(self, *vectors):
        err_state, dim = self.exception_handle(vectors)
        if err_state == False:
            return sum([abs(vectors[0][d]-vectors[1][d]) for d in range(dim)])
    
    def Chebyshev_distance(self, *vectors):
        err_state, dim = self.exception_handle(vectors)
        if err_state == False:
            return max([abs(vectors[0][d]-vectors[1][d]) for d in range(dim)])
    
    def Cosine_similarity(self, *vectors):
        err_state, dim = self.exception_handle(vectors)
        if err_state == False:
            dot_product = sum([vectors[0][d] * vectors[1][d] for d in range(dim)])
            length_of_vec = lambda nums_in_vec: np.sqrt(sum([num**2 for num in nums_in_vec]))
            return dot_product / (length_of_vec(vectors[0]) * length_of_vec(vectors[1]))
    
    # a metric to evaluate similarity of 2 sets
    def Jaccard_similarity_coefficient(self, *vectors):
        err_state, dim = self.exception_handle(vectors)
        if err_state == False:
            set_A, set_B = set(vectors[0]), set(vectors[1])
            #print(set_A, set_B)
            return len(set_A & set_B) / len(set_A | set_B)
    # opposite of Jaccard_similarity_coefficient, to measure the distinction of 2 sets
    def Jaccard_similarity_distance(self, *vectors):
        err_state, dim = self.exception_handle(vectors)
        if err_state == False:
            set_A, set_B = set(vectors[0]), set(vectors[1])
            #print(set_A, set_B)
            return 1 - len(set_A & set_B) / len(set_A | set_B)
    
if __name__ == "__main__":
    sim = Similarity()
    print("Euclidean distance:")
    print("Set 1: ", sim.Euclidean_distance((2,3),(9,17)))
    print("Set 2: ", sim.Euclidean_distance((2,3,4),(9,17,23)))
    
    print('\n', "Manhattan distance", sep='')
    print("Set 1: ", sim.Manhattan_distance((2,3),(9,17)))
    print("Set 2: ", sim.Manhattan_distance((2,3,4),(9,17,23)))
    
    print('\n', "Chebyshev distance", sep='')
    print("Set 1: ", sim.Chebyshev_distance((2,3),(9,17)))
    print("Set 2: ", sim.Chebyshev_distance((2,3,4),(9,17,23)))
    
    print('\n', "Cosine similarity", sep='')
    print("Set 1: ", sim.Cosine_similarity((2,3),(9,17)))
    print("Set 2: ", sim.Cosine_similarity((2,3,4),(9,17,23)))
    print("Set 3: ", sim.Cosine_similarity((2,3,4),(119,17,23)))
    
    print('\n', "Jaccard similarity coefficient", sep='')
    print("Set 1: ", sim.Jaccard_similarity_coefficient((2,3),(9,17)))
    print("Set 2: ", sim.Jaccard_similarity_coefficient((2,9),(9,17)))
    print("Set 3: ", sim.Jaccard_similarity_coefficient((2,3,4,9,17),(9,17,3,6,4)))
    
    print('\n', "Jaccard similarity distance", sep='')
    print("Set 1: ", sim.Jaccard_similarity_distance((2,3),(9,17)))
    print("Set 2: ", sim.Jaccard_similarity_distance((2,9),(9,17)))
    print("Set 3: ", sim.Jaccard_similarity_distance((2,3,4,9,17),(9,17,3,6,4)))
