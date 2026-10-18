import { useTranslation } from "react-i18next";
import { Globe } from "lucide-react";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export default function LanguageToggle() {
  const { i18n } = useTranslation();
  const current = i18n.language.startsWith("zh") ? "zh" : "en";

  const switchTo = (lang: string) => {
    void i18n.changeLanguage(lang);
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          className="relative flex items-center gap-1.5 rounded-lg border border-ob-border bg-ob-surface px-3 py-2 text-xs text-ob-muted transition-colors before:absolute before:-inset-2 before:content-[''] hover:border-nebula/50 hover:text-ob-text"
        >
          <Globe className="size-3.5" />
          <span>{current === "zh" ? "中文" : "EN"}</span>
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="end"
        sideOffset={6}
        className="min-w-[120px] border-ob-border bg-ob-surface"
      >
        <DropdownMenuRadioGroup value={current} onValueChange={switchTo}>
          {/* No flag emoji here: Windows does not render regional-indicator
              pairs as flags, so these appeared as the literal letters "CN"/"GB".
              The radio item already draws its own check indicator and the labels
              are self-describing. */}
          <DropdownMenuRadioItem
            value="zh"
            className="cursor-pointer text-sm text-ob-muted focus:text-ob-text data-[state=checked]:text-nebula-light"
          >
            中文
          </DropdownMenuRadioItem>
          <DropdownMenuRadioItem
            value="en"
            className="cursor-pointer text-sm text-ob-muted focus:text-ob-text data-[state=checked]:text-nebula-light"
          >
            English
          </DropdownMenuRadioItem>
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
