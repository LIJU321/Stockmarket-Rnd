from django.db import models

# Create your models here.



class Item(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()

    def __str__(self):
        print("hhh",self.name)
        return self.name
   
item = Item(name="Toy", description="A fun plaything")
print(item)  # Output: hhh Toy
