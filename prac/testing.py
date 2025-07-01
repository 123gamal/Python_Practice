class Animal:
    def bark(self):
        print("this animal can eat")


class Dog:
    def bark(self):
        print("this dog can bark")


class Cat(Animal, Dog):
    def meow(self):
        print("this cat can meow")


cat = Cat()
print((cat.bark()))
