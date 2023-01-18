class IncompleteCoursesError(Exception):
    """Error raised when his/her studies is incomplete"""

    def __init__(self, items, message='Missing item.') -> None:
        self.missing_items = items
        self.message = f'Missing item from input: {", ".join(items)}' if len(
            self.missing_items) > 0 else message
        super().__init__(self.message)


class ProgramNotFoundError(Exception):
    """Error raised when AI model of a  program not found"""
