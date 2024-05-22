from Api.classes import uniq_seq

class Location:
    __slot__ = ('title', 'position', 'icon','link' )

    def __init__(self, title, position, icon, link):
        self.id: str = next(uniq_seq)
        self.title: str  = title
        self.position: list[str] = position
        self.icon: str = icon
        self.link:'Frame' = link

    
    def __repr__(self) -> str:
        return ("[Location: id:{}\n, title:{}\n,"
                " position:{}\n, icon:{}\n, link:{}\n]"
                .format(self.id, self.title,self.position,
                         self.icon, self.link))
    


    
