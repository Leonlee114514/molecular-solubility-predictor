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
          className="flex items-center gap-1.5 rounded-xl border border-ob-border/60 bg-ob-surface/60 px-3 py-2 text-xs text-ob-muted backdrop-blur-xl transition-all hover:border-nebula/50 hover:text-ob-text hover:shadow-glow/30"
        >
          <Globe className="size-3.5" />
          <span>{current === "zh" ? "中文" : "EN"}</span>
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="end"
        sideOffset={6}
        className="min-w-[120px] border-ob-border/50 bg-ob-surface/60 backdrop-blur-2xl"
      >
        <DropdownMenuRadioGroup value={current} onValueChange={switchTo}>
          <DropdownMenuRadioItem
            value="zh"
            className="cursor-pointer text-sm text-ob-muted focus:text-ob-text data-[state=checked]:text-nebula-light"
          >
            <span className="mr-2">🇨🇳</span> 中文
          </DropdownMenuRadioItem>
          <DropdownMenuRadioItem
            value="en"
            className="cursor-pointer text-sm text-ob-muted focus:text-ob-text data-[state=checked]:text-nebula-light"
          >
            <span className="mr-2">🇬🇧</span> English
          </DropdownMenuRadioItem>
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
